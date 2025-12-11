# ComfyUI Loader Utils - Optimized Model Loading

## The Problem: Native ComfyUI Loading and VRAM OOM Issues

ComfyUI's native loading mechanism has a significant limitation that affects users with limited VRAM: **all models are loaded into VRAM before the workflow begins execution**. This approach causes several issues:

- **VRAM Overflow**: All models are loaded simultaneously, consuming excessive GPU memory
- **Unnecessary Loading**: Even models not needed for the current execution are loaded
- **OOM Errors**: Users with limited VRAM experience crashes and out-of-memory errors
- **Inefficient Resource Management**: No control over the loading order or timing

## The Solution: Loader Nodes with "Any" Parameter

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
- **Avoid OOM Errors**: Control when memory-intensive models are loaded
- **Improved Workflow Stability**: More predictable memory usage

## Usage

The "_Any" suffix nodes can be used exactly like their original counterparts, with the added benefit that they can accept connections from any node type via the optional "any" parameter. This enables better workflow design for memory-constrained environments.

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