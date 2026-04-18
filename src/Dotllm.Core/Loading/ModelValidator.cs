using Dotllm.Models;

namespace Dotllm.Loading;

public sealed class ModelValidationResult
{
    public List<string> Errors { get; } = [];
    public List<string> Warnings { get; } = [];
    public bool IsValid => Errors.Count == 0;
}

internal static class ModelValidator
{
    public static ModelValidationResult Validate(GgufModel model, TransformerConfig config)
    {
        var result = new ModelValidationResult();
        var tensorNames = new HashSet<string>(
            model.TensorInfos.Select(t => t.Name), StringComparer.OrdinalIgnoreCase);

        // Required for all architectures
        if (!tensorNames.Contains("token_embd.weight"))
            result.Errors.Add("Missing required tensor: token_embd.weight");

        if (!tensorNames.Contains("output_norm.weight") && !tensorNames.Contains("token_embd_norm.weight"))
            result.Errors.Add("Missing required tensor: output_norm.weight or token_embd_norm.weight");

        if (config.TiedEmbeddings && !tensorNames.Contains("output.weight"))
            result.Warnings.Add("Model uses tied embeddings (no output.weight tensor)");

        // Per-layer validation
        for (var layer = 0; layer < config.LayerCount; layer++)
        {
            var isAttention = config.LayerTypes.Length > layer && config.LayerTypes[layer] == LayerType.Attention;

            // Attention norm
            if (!tensorNames.Contains($"blk.{layer}.attn_norm.weight"))
                result.Errors.Add($"Missing tensor: blk.{layer}.attn_norm.weight");

            if (isAttention)
            {
                // Attention weights
                var hasQ = tensorNames.Contains($"blk.{layer}.attn_q.weight");
                var hasQkv = tensorNames.Contains($"blk.{layer}.attn_qkv.weight");
                if (!hasQ && !hasQkv)
                    result.Errors.Add($"Missing attention weights for layer {layer}: need attn_q.weight or attn_qkv.weight");

                if (!tensorNames.Contains($"blk.{layer}.attn_output.weight"))
                    result.Errors.Add($"Missing tensor: blk.{layer}.attn_output.weight");
            }

            if (config.HasConvLayers && !isAttention)
            {
                var hasConv1d = tensorNames.Contains($"blk.{layer}.conv1d.weight");
                var hasShortconv = tensorNames.Contains($"blk.{layer}.shortconv.conv.weight");
                if (!hasConv1d && !hasShortconv)
                    result.Errors.Add($"Missing conv weight for layer {layer}: need conv1d.weight or shortconv.conv.weight");

                if (!tensorNames.Contains($"blk.{layer}.shortconv.in_proj.weight"))
                    result.Errors.Add($"Missing tensor: blk.{layer}.shortconv.in_proj.weight");
            }

            // FFN weights
            if (!tensorNames.Contains($"blk.{layer}.ffn_up.weight"))
                result.Errors.Add($"Missing tensor: blk.{layer}.ffn_up.weight");
            if (!tensorNames.Contains($"blk.{layer}.ffn_down.weight"))
                result.Errors.Add($"Missing tensor: blk.{layer}.ffn_down.weight");
        }

        // MoE validation
        if (config.ExpertCount > 0)
        {
            var hasMoeGate = false;
            for (var layer = 0; layer < config.LayerCount; layer++)
            {
                if (tensorNames.Contains($"blk.{layer}.ffn_gate_inp.weight"))
                {
                    hasMoeGate = true;
                    break;
                }
            }
            if (!hasMoeGate)
                result.Warnings.Add("MoE config (ExpertCount > 0) but no ffn_gate_inp.weight tensors found");
        }

        return result;
    }
}
