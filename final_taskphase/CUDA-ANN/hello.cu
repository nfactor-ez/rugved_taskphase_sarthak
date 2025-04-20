#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h> 
#include <math.h>
#include <time.h>
#include <string.h>


#define CUDA_CHECK(err) if (err != cudaSuccess) { fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); exit(1); }


typedef struct {
    float *data; int *labels;
    int n_samples, n_features, n_classes;
} Dataset;

typedef struct {
    float *gamma, *beta, *d_gamma, *d_beta, *running_mean, *running_var;
    float epsilon, momentum;
} BatchNormParams;

typedef struct {
    int size; float *data, *velocity;
} LayerParams;

typedef struct {
    int n_layers, *layer_sizes;
    LayerParams *weights, *biases;
    float **activations, **z_values, **deltas;
    BatchNormParams **bn_params;
    cublasHandle_t cublas_handle;
    curandGenerator_t curand_gen;
} NeuralNetwork;

__global__ void Relu(float *z, float *a, int n, float alpha, bool forward) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (forward) a[idx] = z[idx] > 0 ? z[idx] : alpha * z[idx];
        else a[idx] *= (z[idx] > 0) ? 1.0f : alpha;
    }
}

__global__ void softmaxForward(float *z, float *a, int n, int batch_size) {
    int batch_idx = blockIdx.y;
    if (batch_idx < batch_size) {
        float *batch_z = z + batch_idx * n, *batch_a = a + batch_idx * n;
        float max_val = batch_z[0], sum = 0.0f;
        for (int i = 1; i < n; i++) max_val = fmaxf(max_val, batch_z[i]);
        for (int i = 0; i < n; i++) { batch_a[i] = expf(batch_z[i] - max_val); sum += batch_a[i]; }
        for (int i = 0; i < n; i++) batch_a[i] /= sum;
    }
}

__global__ void batchNorm(float *input, float *output, float *gamma, float *beta, float *running_mean, float *running_var,
                          float *d_output, float *d_input, float *d_gamma, float *d_beta, int batch_size, int features,
                          float epsilon, float momentum, bool training, bool forward) {
    int f = blockIdx.x;
    if (f >= features) return;
    float mean = 0.0f, var = 0.0f, stddev_inv;
    if (training && forward) {
        for (int i = 0; i < batch_size; i++) mean += input[i * features + f];
        mean /= batch_size;
        for (int i = 0; i < batch_size; i++) {
            float diff = input[i * features + f] - mean;
            var += diff * diff;
        }
        var /= batch_size;
        running_mean[f] = momentum * running_mean[f] + (1.0f - momentum) * mean;
        running_var[f] = momentum * running_var[f] + (1.0f - momentum) * var;
        stddev_inv = 1.0f / sqrtf(var + epsilon);
        for (int i = 0; i < batch_size; i++) {
            float norm = (input[i * features + f] - mean) * stddev_inv;
            output[i * features + f] = gamma[f] * norm + beta[f];
        }
    } else if (forward) {
        stddev_inv = 1.0f / sqrtf(running_var[f] + epsilon);
        for (int i = 0; i < batch_size; i++) {
            float norm = (input[i * features + f] - running_mean[f]) * stddev_inv;
            output[i * features + f] = gamma[f] * norm + beta[f];
        }
    } else {
        for (int i = 0; i < batch_size; i++) mean += input[i * features + f];
        mean /= batch_size;
        for (int i = 0; i < batch_size; i++) {
            float diff = input[i * features + f] - mean;
            var += diff * diff;
        }
        var /= batch_size;
        stddev_inv = 1.0f / sqrtf(var + epsilon);
        float d_gamma_val = 0.0f, d_beta_val = 0.0f, d_var = 0.0f, d_mean = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float norm = (input[i * features + f] - mean) * stddev_inv;
            d_gamma_val += d_output[i * features + f] * norm;
            d_beta_val += d_output[i * features + f];
        }
        d_gamma[f] = d_gamma_val; d_beta[f] = d_beta_val;
        for (int i = 0; i < batch_size; i++)
            d_var += d_output[i * features + f] * gamma[f] * (input[i * features + f] - mean) * (-0.5f) * powf(var + epsilon, -1.5f);
        for (int i = 0; i < batch_size; i++) {
            d_mean += d_output[i * features + f] * gamma[f] * (-stddev_inv);
            d_mean += d_var * (-2.0f) * (input[i * features + f] - mean) / batch_size;
        }
        for (int i = 0; i < batch_size; i++) {
            d_input[i * features + f] = d_output[i * features + f] * gamma[f] * stddev_inv +
                                        d_var * 2.0f * (input[i * features + f] - mean) / batch_size +
                                        d_mean / batch_size;
        }
    }
}

__global__ void dropoutForward(float *input, float *output, float *mask, int n, float p, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && p > 0.0f) {
        curandState state; curand_init(seed, idx, 0, &state);
        mask[idx] = (curand_uniform(&state) > p) ? 1.0f : 0.0f;
        output[idx] = input[idx] * mask[idx] / (1.0f - p);
    } else if (idx < n) output[idx] = input[idx];
}

__global__ void crossEntropyLoss(float *predictions, int *labels, float *loss, float *grad_output, int batch_size, int n_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int label = labels[idx];
        atomicAdd(loss, -logf(fmaxf(predictions[idx * n_classes + label], 1e-15f)));
        for (int c = 0; c < n_classes; c++)
            grad_output[idx * n_classes + c] = predictions[idx * n_classes + c] - (c == label ? 1.0f : 0.0f);
    }
}

// Data loading
Dataset loadDataset(const char *data_file, const char *label_file, int n_samples, int n_features, int n_classes) {
    Dataset dataset = {NULL, NULL, n_samples, n_features, n_classes};
    dataset.data = (float *)malloc(n_samples * n_features * sizeof(float));
    dataset.labels = (int *)malloc(n_samples * sizeof(int));
    if (!dataset.data || !dataset.labels) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }

    FILE *file = fopen(data_file, "rb");
    if (!file || fread(dataset.data, n_samples * n_features * sizeof(float), 1, file) != 1) {
        fprintf(stderr, "Failed to load %s\n", data_file); free(dataset.data); free(dataset.labels); exit(1);
    }
    fclose(file);

    file = fopen(label_file, "rb");
    if (!file) { fprintf(stderr, "Failed to load %s\n", label_file); free(dataset.data); free(dataset.labels); exit(1); }
    uint8_t *temp = (uint8_t *)malloc(n_samples);
    if (fread(temp, 1, n_samples, file) != n_samples) {
        fprintf(stderr, "Failed to read labels\n"); free(temp); free(dataset.data); free(dataset.labels); exit(1);
    }
    for (int i = 0; i < n_samples; i++) dataset.labels[i] = temp[i];
    free(temp); fclose(file);

    return dataset;
}

void normalizeData(float *data, int n_samples, int n_features) {
    for (int j = 0; j < n_features; j++) {
        float mean = 0.0f, std = 0.0f;
        for (int i = 0; i < n_samples; i++) mean += data[i * n_features + j];
        mean /= n_samples;
        for (int i = 0; i < n_samples; i++) std += powf(data[i * n_features + j] - mean, 2);
        std = sqrtf(std / n_samples) > 1e-5 ? sqrtf(std / n_samples) : 1.0f;
        for (int i = 0; i < n_samples; i++) data[i * n_features + j] = (data[i * n_features + j] - mean) / std;
    }
}

// Neural network initialization
BatchNormParams *initBatchNormParams(int size) {
    BatchNormParams *params = (BatchNormParams *)malloc(sizeof(BatchNormParams));
    CUDA_CHECK(cudaMalloc(&params->gamma, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params->beta, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params->d_gamma, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params->d_beta, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params->running_mean, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&params->running_var, size * sizeof(float)));

    float *h_gamma = (float *)calloc(size, sizeof(float)), *h_beta = (float *)calloc(size, sizeof(float));
    for (int i = 0; i < size; i++) h_gamma[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(params->gamma, h_gamma, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(params->beta, h_beta, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(params->d_gamma, 0, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(params->d_beta, 0, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(params->running_mean, 0, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(params->running_var, 0, size * sizeof(float)));
    free(h_gamma); free(h_beta);

    params->epsilon = 1e-5; params->momentum = 0.9;
    return params;
}

NeuralNetwork initNeuralNetwork(int n_layers, int *layer_sizes) {
    NeuralNetwork nn = {n_layers, NULL};
    nn.layer_sizes = (int *)malloc(n_layers * sizeof(int));
    memcpy(nn.layer_sizes, layer_sizes, n_layers * sizeof(int));
    nn.weights = (LayerParams *)malloc((n_layers - 1) * sizeof(LayerParams));
    nn.biases = (LayerParams *)malloc((n_layers - 1) * sizeof(LayerParams));
    nn.activations = (float **)calloc(n_layers, sizeof(float *));
    nn.z_values = (float **)calloc(n_layers - 1, sizeof(float *));
    nn.deltas = (float **)calloc(n_layers - 1, sizeof(float *));
    nn.bn_params = (BatchNormParams **)malloc((n_layers - 1) * sizeof(BatchNormParams *));
    if (!nn.weights || !nn.biases || !nn.activations || !nn.z_values || !nn.deltas || !nn.bn_params || !nn.layer_sizes)
        { fprintf(stderr, "Memory allocation failed\n"); exit(1); }

    cublasCreate_v2(&nn.cublas_handle);
    curandCreateGenerator(&nn.curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(nn.curand_gen, time(NULL));

    for (int i = 0; i < n_layers - 1; i++) {
        int m = layer_sizes[i], n = layer_sizes[i + 1];
        nn.weights[i].size = m * n; nn.biases[i].size = n;
        CUDA_CHECK(cudaMalloc(&nn.weights[i].data, m * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn.biases[i].data, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn.weights[i].velocity, m * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn.biases[i].velocity, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(nn.weights[i].velocity, 0, m * n * sizeof(float)));
        CUDA_CHECK(cudaMemset(nn.biases[i].velocity, 0, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(nn.biases[i].data, 0, n * sizeof(float)));
        curandGenerateNormal(nn.curand_gen, nn.weights[i].data, m * n, 0.0f, sqrtf(2.0f / m));
        nn.bn_params[i] = initBatchNormParams(n);
    }
    return nn;
}


void forwardPropagate(NeuralNetwork nn, float *input, int batch_size, bool training, unsigned long seed) {
    for (int i = 0; i < nn.n_layers - 1; i++) {
        CUDA_CHECK(cudaMalloc(&nn.activations[i], nn.layer_sizes[i] * batch_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn.z_values[i], nn.layer_sizes[i + 1] * batch_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn.deltas[i], nn.layer_sizes[i + 1] * batch_size * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&nn.activations[nn.n_layers - 1], nn.layer_sizes[nn.n_layers - 1] * batch_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(nn.activations[0], input, nn.layer_sizes[0] * batch_size * sizeof(float), cudaMemcpyHostToDevice));

    float dropout_rate = training ? 0.3 : 0.0;
    for (int i = 0; i < nn.n_layers - 1; i++) {
        int m = nn.layer_sizes[i], n = nn.layer_sizes[i + 1];
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm_v2(nn.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, batch_size, m, &alpha,
                       nn.weights[i].data, m, nn.activations[i], m, &beta, nn.z_values[i], n);
        for (int j = 0; j < batch_size; j++)
            cublasSaxpy_v2(nn.cublas_handle, n, &alpha, nn.biases[i].data, 1, nn.z_values[i] + j * n, 1);

        if (i < nn.n_layers - 2) {
            float *norm_out; CUDA_CHECK(cudaMalloc(&norm_out, n * batch_size * sizeof(float)));
            batchNorm<<<dim3(n), 1>>>(nn.z_values[i], norm_out, nn.bn_params[i]->gamma, nn.bn_params[i]->beta,
                                      nn.bn_params[i]->running_mean, nn.bn_params[i]->running_var, NULL, NULL, NULL, NULL,
                                      batch_size, n, nn.bn_params[i]->epsilon, nn.bn_params[i]->momentum, training, true);
            Relu<<<dim3((n * batch_size + 255) / 256), 256>>>(norm_out, nn.activations[i + 1], n * batch_size, 0.01, true);
            if (dropout_rate > 0.0) {
                float *mask; CUDA_CHECK(cudaMalloc(&mask, n * batch_size * sizeof(float)));
                dropoutForward<<<dim3((n * batch_size + 255) / 256), 256>>>(nn.activations[i + 1], nn.activations[i + 1], mask, n * batch_size, dropout_rate, seed);
                CUDA_CHECK(cudaFree(mask));
            }
            CUDA_CHECK(cudaFree(norm_out));
        } else {
            softmaxForward<<<dim3(1, batch_size), 32>>>(nn.z_values[i], nn.activations[i + 1], n, batch_size);
        }
    }
}

void backwardPropagate(NeuralNetwork nn, float *input, int *labels_device, int batch_size, float lr, unsigned long seed) {
    int out_size = nn.layer_sizes[nn.n_layers - 1];
    float *loss, *grad_out;
    CUDA_CHECK(cudaMalloc(&loss, sizeof(float))); CUDA_CHECK(cudaMemset(loss, 0, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_out, batch_size * out_size * sizeof(float)));
    crossEntropyLoss<<<dim3((batch_size + 255) / 256), 256>>>(nn.activations[nn.n_layers - 1], labels_device, loss, grad_out, batch_size, out_size);
    CUDA_CHECK(cudaMemcpy(nn.deltas[nn.n_layers - 2], grad_out, out_size * batch_size * sizeof(float), cudaMemcpyDeviceToDevice));

    float momentum = 0.9;
    for (int i = nn.n_layers - 2; i >= 0; i--) {
        int curr_size = nn.layer_sizes[i + 1], prev_size = nn.layer_sizes[i];
        if (i < nn.n_layers - 2) {
            float *d_norm; CUDA_CHECK(cudaMalloc(&d_norm, curr_size * batch_size * sizeof(float)));
            Relu<<<dim3((batch_size * curr_size + 255) / 256), 256>>>(nn.z_values[i], nn.deltas[i], curr_size * batch_size, 0.01, false);
            batchNorm<<<dim3(curr_size), 1>>>(nn.z_values[i], NULL, nn.bn_params[i]->gamma, nn.bn_params[i]->beta,
                                              nn.bn_params[i]->running_mean, nn.bn_params[i]->running_var, nn.deltas[i], d_norm,
                                              nn.bn_params[i]->d_gamma, nn.bn_params[i]->d_beta, batch_size, curr_size,
                                              nn.bn_params[i]->epsilon, nn.bn_params[i]->momentum, true, false);
            float alpha = -lr;
            cublasSaxpy_v2(nn.cublas_handle, curr_size, &alpha, nn.bn_params[i]->d_gamma, 1, nn.bn_params[i]->gamma, 1);
            cublasSaxpy_v2(nn.cublas_handle, curr_size, &alpha, nn.bn_params[i]->d_beta, 1, nn.bn_params[i]->beta, 1);
            CUDA_CHECK(cudaMemcpy(nn.deltas[i], d_norm, curr_size * batch_size * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaFree(d_norm));
        }

        float alpha = 1.0f / batch_size, beta = 1.0f, neg_lr = -lr;
        cublasSscal_v2(nn.cublas_handle, prev_size * curr_size, &momentum, nn.weights[i].velocity, 1);
        cublasSgemm_v2(nn.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, prev_size, curr_size, batch_size, &alpha,
                       nn.activations[i], prev_size, nn.deltas[i], curr_size, &beta, nn.weights[i].velocity, prev_size);
        cublasSaxpy_v2(nn.cublas_handle, prev_size * curr_size, &neg_lr, nn.weights[i].velocity, 1, nn.weights[i].data, 1);

        alpha = -lr / batch_size;
        cublasSscal_v2(nn.cublas_handle, curr_size, &momentum, nn.biases[i].velocity, 1);
        for (int j = 0; j < batch_size; j++)
            cublasSaxpy_v2(nn.cublas_handle, curr_size, &alpha, nn.deltas[i] + j * curr_size, 1, nn.biases[i].velocity, 1);
        cublasSaxpy_v2(nn.cublas_handle, curr_size, &neg_lr, nn.biases[i].velocity, 1, nn.biases[i].data, 1);

        if (i > 0) {
            alpha = 1.0f; beta = 0.0f;
            cublasSgemm_v2(nn.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, prev_size, batch_size, curr_size, &alpha,
                           nn.weights[i].data, prev_size, nn.deltas[i], curr_size, &beta, nn.deltas[i - 1], prev_size);
        }
    }

    for (int i = 0; i < nn.n_layers - 1; i++) {
        CUDA_CHECK(cudaFree(nn.activations[i])); CUDA_CHECK(cudaFree(nn.z_values[i])); CUDA_CHECK(cudaFree(nn.deltas[i]));
    }
    CUDA_CHECK(cudaFree(nn.activations[nn.n_layers - 1])); CUDA_CHECK(cudaFree(loss)); CUDA_CHECK(cudaFree(grad_out));
}


float calculateAccuracy(NeuralNetwork nn, float *data, int *labels, int n_samples) {
    int batch_size = 100, correct = 0;
    float *output = (float *)malloc(batch_size * nn.layer_sizes[nn.n_layers - 1] * sizeof(float));
    for (int i = 0; i < n_samples; i += batch_size) {
        int curr_batch = fmin(batch_size, n_samples - i);
        float *batch_data; CUDA_CHECK(cudaMalloc(&batch_data, curr_batch * nn.layer_sizes[0] * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(batch_data, data + i * nn.layer_sizes[0], curr_batch * nn.layer_sizes[0] * sizeof(float), cudaMemcpyHostToDevice));
        int *batch_labels; CUDA_CHECK(cudaMalloc(&batch_labels, curr_batch * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(batch_labels, labels + i, curr_batch * sizeof(int), cudaMemcpyHostToDevice));

        forwardPropagate(nn, batch_data, curr_batch, false, time(NULL));
        CUDA_CHECK(cudaMemcpy(output, nn.activations[nn.n_layers - 1], curr_batch * nn.layer_sizes[nn.n_layers - 1] * sizeof(float), cudaMemcpyDeviceToHost));

        for (int j = 0; j < curr_batch; j++) {
            int pred = 0; float max_prob = output[j * nn.layer_sizes[nn.n_layers - 1]];
            for (int k = 1; k < nn.layer_sizes[nn.n_layers - 1]; k++)
                if (output[j * nn.layer_sizes[nn.n_layers - 1] + k] > max_prob) {
                    max_prob = output[j * nn.layer_sizes[nn.n_layers - 1] + k]; pred = k;
                }
            if (pred == labels[i + j]) correct++;
        }

        for (int k = 0; k < nn.n_layers - 1; k++) {
            CUDA_CHECK(cudaFree(nn.activations[k])); CUDA_CHECK(cudaFree(nn.z_values[k])); CUDA_CHECK(cudaFree(nn.deltas[k]));
        }
        CUDA_CHECK(cudaFree(nn.activations[nn.n_layers - 1])); CUDA_CHECK(cudaFree(batch_data)); CUDA_CHECK(cudaFree(batch_labels));
    }
    free(output);
    return (float)correct / n_samples;
}

void freeNeuralNetwork(NeuralNetwork nn) {
    for (int i = 0; i < nn.n_layers - 1; i++) {
        CUDA_CHECK(cudaFree(nn.weights[i].data)); CUDA_CHECK(cudaFree(nn.biases[i].data));
        CUDA_CHECK(cudaFree(nn.weights[i].velocity)); CUDA_CHECK(cudaFree(nn.biases[i].velocity));
        CUDA_CHECK(cudaFree(nn.bn_params[i]->gamma)); CUDA_CHECK(cudaFree(nn.bn_params[i]->beta));
        CUDA_CHECK(cudaFree(nn.bn_params[i]->d_gamma)); CUDA_CHECK(cudaFree(nn.bn_params[i]->d_beta));
        CUDA_CHECK(cudaFree(nn.bn_params[i]->running_mean)); CUDA_CHECK(cudaFree(nn.bn_params[i]->running_var));
        free(nn.bn_params[i]);
    }
    free(nn.weights); free(nn.biases); free(nn.activations); free(nn.z_values); free(nn.deltas); free(nn.bn_params); free(nn.layer_sizes);
    cublasDestroy(nn.cublas_handle); curandDestroyGenerator(nn.curand_gen);
}

void shuffleDataset(Dataset dataset) {
    for (int i = dataset.n_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        for (int k = 0; k < dataset.n_features; k++) {
            float temp = dataset.data[i * dataset.n_features + k];
            dataset.data[i * dataset.n_features + k] = dataset.data[j * dataset.n_features + k];
            dataset.data[j * dataset.n_features + k] = temp;
        }
        int temp = dataset.labels[i]; dataset.labels[i] = dataset.labels[j]; dataset.labels[j] = temp;
    }
}


int main() {
    int n_epochs = 10, batch_size =16, layer_sizes[] = {784, 64, 32, 10};
    float lr =0.0005;

    Dataset train = loadDataset("Downloads/x_train2.bin", "Downloads/y_train2.bin", 60000, 784, 10);
    Dataset test = loadDataset("Downloads/x_test2.bin", "Downloads/y_test2.bin", 10000, 784, 10);
    normalizeData(train.data, train.n_samples, train.n_features);
    normalizeData(test.data, test.n_samples, test.n_features);

    NeuralNetwork nn = initNeuralNetwork(4, layer_sizes);
    srand(time(NULL));

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        shuffleDataset(train);
        for (int i = 0; i < train.n_samples; i += batch_size) {
            int curr_batch = fmin(batch_size, train.n_samples - i);
            float *batch_data; CUDA_CHECK(cudaMalloc(&batch_data, curr_batch * train.n_features * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(batch_data, train.data + i * train.n_features, curr_batch * train.n_features * sizeof(float), cudaMemcpyHostToDevice));
            int *batch_labels; CUDA_CHECK(cudaMalloc(&batch_labels, curr_batch * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(batch_labels, train.labels + i, curr_batch * sizeof(int), cudaMemcpyHostToDevice));

            forwardPropagate(nn, batch_data, curr_batch, true, rand());
            backwardPropagate(nn, batch_data, batch_labels, curr_batch, lr, rand());

            CUDA_CHECK(cudaFree(batch_data)); CUDA_CHECK(cudaFree(batch_labels));
        }
    
        printf("Epoch %d: Train acc = %.2f%%\n", epoch + 1,
               calculateAccuracy(nn, train.data, train.labels, train.n_samples) * 100);

    }
    printf(" Test acc = %.2f%%\n",calculateAccuracy(nn, test.data, test.labels, test.n_samples) * 100);


    freeNeuralNetwork(nn); free(train.data); free(train.labels); free(test.data); free(test.labels);
    return 0;
}