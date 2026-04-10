using Dotllm.Models;

namespace Dotllm.Inference;

internal sealed class InferenceBuffers
{
    public readonly float[] HiddenState;
    public readonly float[] Logits;
    public readonly float[] NormBuf;
    public readonly float[] NormBuf2;
    public readonly float[] QBuf;
    public readonly float[] KBuf;
    public readonly float[] VBuf;
    public readonly float[] AttnOutBuf;
    public readonly float[] AttnResultBuf;
    public readonly float[] FfnBuf;
    public readonly float[] FfnResultBuf;
    public readonly float[] ConvBuf;

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
        QBuf = new float[qDim];
        KBuf = new float[kvDim];
        VBuf = new float[kvDim];
        AttnOutBuf = new float[hidden];
        AttnResultBuf = new float[qDim];
        FfnBuf = new float[ffnDim];
        FfnResultBuf = new float[hidden];
        ConvBuf = new float[ffnDim];
    }
}