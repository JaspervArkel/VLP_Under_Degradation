#include <stdio.h>
#include <math.h>
#include <string.h>
#include "pico/stdlib.h"
#include <stdlib.h>
#include "knn_reference_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "normalization_data.h"

constexpr int kTensorArenaSize = 80 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

float last_cleaned_time = 0.0f;
float current_age_hours = 0.0f;
float current_temperature = 0.0f;
bool current_cleaned = false;

void repair_blockage_and_broken(float X[36], float rough_x, float rough_y) {
    float best_dist[KNN_K];
    int best_idx[KNN_K];

    for (int k = 0; k < KNN_K; k++) {
        best_dist[k] = 1e30f;
        best_idx[k] = -1;
    }

    // Find K nearest reference positions to rough prediction
    for (int n = 0; n < KNN_REF_N; n++) {
        float dx = rough_x - KNN_TRAIN_Y[n][0];
        float dy = rough_y - KNN_TRAIN_Y[n][1];
        float dist = dx * dx + dy * dy;

        // Insert into top-K smallest distances
        int worst_k = 0;
        for (int k = 1; k < KNN_K; k++) {
            if (best_dist[k] > best_dist[worst_k]) {
                worst_k = k;
            }
        }

        if (dist < best_dist[worst_k]) {
            best_dist[worst_k] = dist;
            best_idx[worst_k] = n;
        }
    }

    float expected[36];
    for (int i = 0; i < 36; i++) {
        expected[i] = 0.0f;
    }

    float weight_sum = 0.0f;

    for (int k = 0; k < KNN_K; k++) {
        if (best_idx[k] < 0) {
            continue;
        }

        float dist = sqrtf(best_dist[k]);
        float weight = 1.0f / (dist + 1e-6f);
        weight_sum += weight;

        int n = best_idx[k];

        for (int i = 0; i < 36; i++) {
            expected[i] += KNN_TRAIN_X[n][i] * weight;
        }
    }

    if (weight_sum <= 0.0f) {
        return;
    }

    for (int i = 0; i < 36; i++) {
        expected[i] /= weight_sum;
    }

    int replaced = 0;

    for (int i = 0; i < 36; i++) {
        if (expected[i] > 1e-4f) {
            float diff_ratio = X[i] / (expected[i] + 1e-8f);

            if (diff_ratio < 0.25f) {
                X[i] = expected[i];
                replaced++;
            }
        }
    }

    printf("REPAIRED,%d\n", replaced);
    fflush(stdout);
}

void run_model(
    tflite::MicroInterpreter& interpreter,
    TfLiteTensor* input,
    TfLiteTensor* output,
    float X[36],
    float* pred_x,
    float* pred_y
) {
    for (int i = 0; i < 36; i++) {
        int32_t q = (int32_t)roundf(X[i] / input->params.scale)
                    + input->params.zero_point;

        if (q > 127) q = 127;
        if (q < -128) q = -128;

        input->data.int8[i] = (int8_t)q;
    }

    TfLiteStatus invoke_status = interpreter.Invoke();

    if (invoke_status != kTfLiteOk) {
        printf("ERROR invoke failed\n");
        fflush(stdout);
        *pred_x = 0.0f;
        *pred_y = 0.0f;
        return;
    }

    int8_t x_q = output->data.int8[0];
    int8_t y_q = output->data.int8[1];

    *pred_x = ((float)x_q - output->params.zero_point) * output->params.scale;
    *pred_y = ((float)y_q - output->params.zero_point) * output->params.scale;
}

void anti_age(float X[36], float age, float alpha) {
    float factor = powf(1.0f / (1.0f - alpha / (365.0f * 24.0f)), age);

    for (int i = 0; i < 36; i++) {
        X[i] *= factor;
    }
}

void anti_dust(float X[36], bool cleaned, float time) {
    if (cleaned) {
        last_cleaned_time = time;
    }

    float dust_collection_time = time - last_cleaned_time;

    float expected_reduction_sensor =
        powf(1.0f - 0.015f / (730.0f * 24.0f), dust_collection_time);

    float expected_reduction_lights =
        powf(1.0f - 0.08f / (730.0f * 24.0f), dust_collection_time);

    float factor = (1.0f / expected_reduction_lights) *
                   (1.0f / expected_reduction_sensor);

    for (int i = 0; i < 36; i++) {
        X[i] *= factor;
    }
}

void anti_thermal_droop(float X[36], float temperature, float coefficient) {
    float change = 1.0f / (1.0f - coefficient * temperature);

    for (int i = 0; i < 36; i++) {
        X[i] *= change;
    }
}

void anti_degrade(
    float X[36],
    float temperature,
    float age,
    float alpha,
    bool cleaned,
    float thermal_coeff
) {
    anti_age(X, age, alpha);
    anti_dust(X, cleaned, age);
    anti_thermal_droop(X, temperature, thermal_coeff);
}
bool read_measurement(float X[36]) {
    char buffer[1400];

    int index = 0;

    printf("READY\n");
    fflush(stdout);

    while (index < 1399) {
        int c = getchar_timeout_us(1000000); // wait 1 second

        if (c == PICO_ERROR_TIMEOUT) {
            printf("READY\n");
            fflush(stdout);
            continue;
        }

        if (c == '\n' || c == '\r') {
            if (index == 0) {
                continue;
            }
            break;
        }

        buffer[index++] = (char)c;
    }

    buffer[index] = '\0';

    printf("RECEIVED,%s\n", buffer);
    fflush(stdout);

    char* ptr = buffer;
    char* endptr;

    current_age_hours = strtof(ptr, &endptr);
    if (ptr == endptr) {
        printf("ERROR parse age\n");
        fflush(stdout);
        return false;
    }
    ptr = endptr;
    if (*ptr == ',') ptr++;

    current_temperature = strtof(ptr, &endptr);
    if (ptr == endptr) {
        printf("ERROR parse temperature\n");
        fflush(stdout);
        return false;
    }
    ptr = endptr;
    if (*ptr == ',') ptr++;

    float cleaned_float = strtof(ptr, &endptr);
    if (ptr == endptr) {
       printf("ERROR parse cleaned\n");
      fflush(stdout);
      return false;
    }
    current_cleaned = cleaned_float != 0.0f;
    ptr = endptr;
    if (*ptr == ',') ptr++;

    for (int i = 0; i < 36; i++) {
        X[i] = strtof(ptr, &endptr);

        if (ptr == endptr) {
            printf("ERROR parse sensor %d\n", i);
            fflush(stdout);
            return false;
        }

        ptr = endptr;
        if (*ptr == ',') ptr++;
    }

    printf("INPUT_LOADED\n");
    fflush(stdout);

    return true;
}

int main() {
    stdio_init_all();
    sleep_ms(3000);

    printf("Starting Pico TFLite Micro inference\n");
    fflush(stdout);

    const tflite::Model* model = tflite::GetModel(g_model);

    static tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();

    static tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    TfLiteStatus allocate_status = interpreter.AllocateTensors();

    if (allocate_status != kTfLiteOk) {
        printf("ERROR AllocateTensors failed\n");
        fflush(stdout);

        while (true) {
            sleep_ms(1000);
        }
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("Input scale: %f, zero point: %d\n",
           input->params.scale,
           input->params.zero_point);
    fflush(stdout);

    printf("Output scale: %f, zero point: %d\n",
           output->params.scale,
           output->params.zero_point);
    fflush(stdout);

    printf("READY_TO_RECEIVE\n");
    fflush(stdout);

    while (true) {
        float X_raw[36];

        if (!read_measurement(X_raw)) {
            printf("ERROR read_measurement failed\n");
            fflush(stdout);
            continue;
        }

        float temperature = current_temperature;
        float age_hours = current_age_hours;
        bool cleaned = current_cleaned;

        float alpha = 0.03f;
        float thermal_coeff = 0.003f;

        /*
         * Step 1: Apply anti-degradation correction in raw sensor space.
         * For clean baseline testing, you can temporarily comment this out.
         */
        anti_degrade(
            X_raw,
            temperature,
            age_hours,
            alpha,
            cleaned,
            thermal_coeff
        );

        /*
         * Step 2: First inference pass.
         * Normalize raw/corrected input before feeding the int8 model.
         */
        float X_norm[36];

        for (int i = 0; i < 36; i++) {
            X_norm[i] = (X_raw[i] - X_MEAN[i]) / X_STD[i];
        }

        float rough_x = 0.0f;
        float rough_y = 0.0f;

        run_model(
            interpreter,
            input,
            output,
            X_norm,
            &rough_x,
            &rough_y
        );

        printf("ROUGH,%f,%f\n", rough_x, rough_y);
        fflush(stdout);

        /*
         * Step 3: Repair blocked/broken LED values in raw sensor space.
         * This uses rough_x/rough_y to find nearest reference fingerprints.
         */
        float X_repaired[36];

        for (int i = 0; i < 36; i++) {
            X_repaired[i] = X_raw[i];
        }

        repair_blockage_and_broken(
            X_repaired,
            rough_x,
            rough_y
        );

        /*
         * Step 4: Second inference pass on repaired input.
         */
        for (int i = 0; i < 36; i++) {
            X_norm[i] = (X_repaired[i] - X_MEAN[i]) / X_STD[i];
        }

        float pred_x = 0.0f;
        float pred_y = 0.0f;

        run_model(
            interpreter,
            input,
            output,
            X_norm,
            &pred_x,
            &pred_y
        );

        printf("PRED,%f,%f\n", pred_x, pred_y);
        fflush(stdout);

        sleep_ms(50);
    }

    return 0;
}