using Dotllm.Models;

namespace Dotllm.Inference;

internal sealed class InferenceBuffers
{
    public readonly float[] HiddenState;
    public readonly float[] Logits;
    public readonly float[] NormBuf;
    public readonly float[] NormBuf2;
    public readonly float[] NormTempBuf;
    public readonly float[] QBuf;
    public readonly float[] KBuf;
    public readonly float[] VBuf;
    public readonly float[] AttnOutBuf;
    public readonly float[] AttnResultBuf;
    public readonly float[] FfnBuf;
    public readonly float[] FfnResultBuf;
    public readonly float[] ConvBuf;
    public readonly float[] ScoreBuf;
    public readonly float[] SamplingBuf;
    public readonly int[] SamplingIdxBuf;
    public readonly float[] MoeGateLogits;
    public readonly float[] MoeExpertResultBuf;
    public readonly float[] MoeAccumulatorBuf;
    public readonly int[] MoeSelectedExperts;
    public readonly float[] MoeRoutingWeights;

    public InferenceBuffers(TransformerConfig cfg)
    {
        var hidden = cfg.HiddenSize;
        var qDim = cfg.QDim;
        var kvDim = cfg.KvDim;
        var ffnDim = cfg.FfnDim;
        var vocabSize = cfg.VocabSize;

        HiddenState = new float[hidden];
        Logits = new float[vocabSize];
        NormBuf = new float[hidden];
        NormBuf2 = new float[hidden];
        NormTempBuf = new float[hidden];
        QBuf = new float[qDim];
        KBuf = new float[kvDim];
        VBuf = new float[kvDim];
        AttnOutBuf = new float[hidden];
        AttnResultBuf = new float[Math.Max(qDim, hidden)];
        var maxIntermediate = Math.Max(2 * ffnDim, 3 * hidden);
        FfnBuf = new float[maxIntermediate];
        FfnResultBuf = new float[hidden];
        ConvBuf = new float[ffnDim];
        ScoreBuf = new float[Math.Max(hidden, cfg.ContextLength)];
        SamplingBuf = new float[vocabSize];
        SamplingIdxBuf = new int[vocabSize];

        var expertCount = cfg.ExpertCount;
        var topK = cfg.ExpertUsedCount;
        MoeGateLogits = expertCount > 0 ? new float[expertCount] : [];
        MoeExpertResultBuf = expertCount > 0 ? new float[hidden] : [];
        MoeAccumulatorBuf = expertCount > 0 ? new float[hidden] : [];
        MoeSelectedExperts = topK > 0 ? new int[topK] : [];
        MoeRoutingWeights = topK > 0 ? new float[topK] : [];
    }
}