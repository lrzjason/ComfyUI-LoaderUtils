# ComfyUI Loader Utils - Adjust Model Loading Order

## The Problem: Comfyui load models at the start at once

Solution: Added an optional "Any" Parameter to loader node

This custom loader module addresses these issues by:

1. **Flexible Node Connections**: Added an optional "any" parameter to all loader nodes, allowing them to connect to any output type
2. **Controlled Loading Order**: Users can strategically place loader nodes after other nodes, optimizing the model loading sequence
3. **Memory Management**: Enables better VRAM management by controlling when and which models are loaded
4. **Sequential Loading**: Models are loaded only when needed, in a controlled sequence

## Features

- All standard ComfyUI loader nodes included with "_Any" suffix
- Optional "any" parameter for flexible connections
- Maintains all original functionality and parameters
- Compatible with existing ComfyUI workflows

## Available Loader Nodes

- `CheckpointLoader_Any` - Advanced checkpoint loading
- `CheckpointLoaderSimple_Any` - Simple checkpoint loading  
- `DiffusersLoader_Any` - Diffusers model loading
- `unCLIPCheckpointLoader_Any` - unCLIP checkpoint loading
- `LoraLoader_Any` - LoRA model loading
- `LoraLoaderModelOnly_Any` - LoRA model only loading
- `VAELoader_Any` - VAE model loading
- `ControlNetLoader_Any` - ControlNet model loading
- `DiffControlNetLoader_Any` - Diffusion ControlNet loading
- `UNETLoader_Any` - UNET model loading
- `CLIPLoader_Any` - CLIP model loading
- `DualCLIPLoader_Any` - Dual CLIP model loading
- `CLIPVisionLoader_Any` - CLIP vision model loading
- `StyleModelLoader_Any` - Style model loading
- `GLIGENLoader_Any` - GLIGEN model loading

## Benefits for Low VRAM Users

- **Reduced Memory Footprint**: Load models only when needed
- **Flexible Sequencing**: Arrange loading order based on available memory
- **Improved Workflow Stability**: More predictable memory usage

## Usage

The "_Any" suffix nodes can be used exactly like their original counterparts, with the added benefit that they can accept connections from any node type via the optional "any" parameter. This enables better workflow design for memory-constrained environments.

## Example

Here's an example workflow showing how the loader nodes with "any" parameter can be used to optimize memory management:

![Example Workflow](example.png)

The workflow file is also available as `workflow.json` in this repository.

## Memory Management Benefits

The key advantage of these loader nodes is that you can control WHEN models are loaded by connecting them strategically in your workflow. In the example above:

1. The UNETLoader_Any is connected after the CLIPTextEncode nodes, allowing them to run before the heavy UNET model is loaded
2. The VAELoader_Any is connected after sampling, allowing you to load the VAE only when needed for decoding

This approach prevents all models from being loaded simultaneously, which was the root cause of VRAM OOM errors in standard ComfyUI usage.

Simply use these nodes in place of the standard loader nodes, and strategically connect them to control when models are loaded into memory.

## Contact
- **Twitter**: [@Lrzjason](https://twitter.com/Lrzjason)
- **Email**: lrzjason@gmail.com
- **QQ Group**: 866612947
- **Wechatid**: fkdeai
- **Civitai**: [xiaozhijason](https://civitai.com/user/xiaozhijason)

## Sponsors me for more open source projects:
<div align="center">
  <table>
    <tr>
      <td align="center">
        <p>Buy me a coffee:</p>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/bmc_qr.png" alt="Buy Me a Coffee QR" width="200" />
      </td>
      <td align="center">
        <p>WeChat:</p>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/wechat.jpg" alt="WeChat QR" width="200" />
      </td>
    </tr>
  </table>
</div>
