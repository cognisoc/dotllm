namespace Llmdot.Inference;

internal sealed class MoeRouter
{
    private readonly IComputeBackend _backend;

    public MoeRouter(IComputeBackend backend)
    {
        _backend = backend;
    }

    public void Route(
        ReadOnlySpan<float> hiddenState,
        ReadOnlySpan<byte> gateWeightData,
        Loading.GgmlType gateType,
        int hiddenSize,
        int expertCount,
        int topK,
        Span<float> gateLogits,
        Span<int> selectedExperts,
        Span<float> routingWeights)
    {
        // 1. MatMul: gateLogits = hiddenState @ gateWeight^T
        _backend.MatMul(hiddenState, gateWeightData, gateLogits, gateType, hiddenSize, expertCount);

        // 2. Find top-K indices via selection sort
        for (var i = 0; i < topK; i++)
        {
            var bestIdx = -1;
            var bestVal = float.MinValue;
            for (var j = 0; j < expertCount; j++)
            {
                var skip = false;
                for (var k = 0; k < i; k++)
                {
                    if (selectedExperts[k] == j) { skip = true; break; }
                }
                if (skip) continue;
                if (gateLogits[j] > bestVal) { bestVal = gateLogits[j]; bestIdx = j; }
            }
            selectedExperts[i] = bestIdx;
            routingWeights[i] = bestVal;
        }

        // 3. Softmax over the selected top-K values
        var maxVal = float.MinValue;
        for (var i = 0; i < topK; i++)
            if (routingWeights[i] > maxVal) maxVal = routingWeights[i];

        var sumExp = 0f;
        for (var i = 0; i < topK; i++)
        {
            routingWeights[i] = MathF.Exp(routingWeights[i] - maxVal);
            sumExp += routingWeights[i];
        }

        for (var i = 0; i < topK; i++)
            routingWeights[i] /= sumExp;
    }
}
