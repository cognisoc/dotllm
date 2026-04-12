// Softmax Metal compute shader (in-place)
// Input/Output: scores[seqLen]
// Uniform: softcap (0 = no softcap), seqLen

#include <metal_stdlib>
using namespace metal;

kernel void softmax(
    device float* scores [[buffer(0)]],
    constant float& softcap [[buffer(1)]],
    constant uint& seqLen [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= seqLen) return;

    // Apply softcap
    if (softcap > 0.0)
        scores[gid] = softcap * tanh(scores[gid] / softcap);

    // Find max
    threadgroup float s_max;
    float thread_max = -3.402823e+38;
    for (uint i = 0; i < seqLen; i++)
        thread_max = max(thread_max, scores[i]);

    s_max = thread_max;
    float max_val = s_max;

    // Exp and normalize
    float thread_sum = 0.0;
    for (uint i = 0; i < seqLen; i++)
    {
        scores[i] = exp(scores[i] - max_val);
        thread_sum += scores[i];
    }

    float inv_sum = 1.0 / thread_sum;
    for (uint i = 0; i < seqLen; i++)
        scores[i] *= inv_sum;
}