use super::stream_image::{
    NativeStreamImage,
    background_shader::{self, Program},
};
use gpui::{
    PaintPostprocess, PostprocessDescriptor, StreamImageBudgets, StreamImageCompletion,
    StreamImageId, StreamImageLease,
};
use std::sync::Arc;
use windows::Win32::Graphics::{
    Direct3D::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, Direct3D11::*, Dxgi::Common::*,
};

struct Target {
    texture: ID3D11Texture2D,
    view: ID3D11ShaderResourceView,
    target: ID3D11RenderTargetView,
}

impl Target {
    fn from_texture(device: &ID3D11Device, texture: ID3D11Texture2D) -> anyhow::Result<Self> {
        let mut view = None;
        let mut target = None;
        unsafe {
            device.CreateShaderResourceView(&texture, None, Some(&mut view))?;
            device.CreateRenderTargetView(&texture, None, Some(&mut target))?;
        }
        Ok(Self {
            texture,
            view: view.ok_or_else(|| anyhow::anyhow!("missing effect input view"))?,
            target: target.ok_or_else(|| anyhow::anyhow!("missing effect output view"))?,
        })
    }
}

struct Resources {
    targets: [Target; 2],
    programs: Vec<Arc<Program>>,
    uniforms: ID3D11Buffer,
    sampler: ID3D11SamplerState,
    _lease: StreamImageLease,
}

pub(super) struct NativePostprocess {
    input: NativeStreamImage,
    descriptor: PostprocessDescriptor,
    context: ID3D11DeviceContext,
    resources: Resources,
}

impl NativePostprocess {
    pub fn new(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        id: StreamImageId,
        budgets: &StreamImageBudgets,
        descriptor: PostprocessDescriptor,
    ) -> anyhow::Result<Self> {
        descriptor.validate()?;
        let extra_bytes = descriptor.texture_bytes()? / 2 + descriptor.uniform_size as u64;
        let lease = budgets.reserve(extra_bytes)?;
        let input =
            NativeStreamImage::new_effect_input(device, context, id, budgets, descriptor.size)?;
        let first = Target::from_texture(device, input.texture().clone())?;
        let mut description = D3D11_TEXTURE2D_DESC::default();
        unsafe { first.texture.GetDesc(&mut description) };
        let mut second = None;
        unsafe { device.CreateTexture2D(&description, None, Some(&mut second)) }?;
        let second = Target::from_texture(
            device,
            second.ok_or_else(|| anyhow::anyhow!("missing effect scratch texture"))?,
        )?;
        let mut uniforms = None;
        let mut sampler = None;
        unsafe {
            device.CreateBuffer(
                &D3D11_BUFFER_DESC {
                    ByteWidth: descriptor.uniform_size as u32,
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                    ..Default::default()
                },
                None,
                Some(&mut uniforms),
            )?;
            device.CreateSamplerState(
                &D3D11_SAMPLER_DESC {
                    Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                    AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                    AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                    AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                    MaxAnisotropy: 1,
                    ComparisonFunc: D3D11_COMPARISON_NEVER,
                    MaxLOD: f32::MAX,
                    ..Default::default()
                },
                Some(&mut sampler),
            )?;
        }
        let programs = descriptor
            .directx_passes
            .iter()
            .map(|code| background_shader::prepare_program(device, code))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            input,
            descriptor,
            context: context.clone(),
            resources: Resources {
                targets: [first, second],
                programs,
                uniforms: uniforms.ok_or_else(|| anyhow::anyhow!("missing effect constants"))?,
                sampler: sampler.ok_or_else(|| anyhow::anyhow!("missing effect sampler"))?,
                _lease: lease,
            },
        })
    }

    pub fn belongs_to_device(&self, device: &ID3D11Device) -> bool {
        self.input.belongs_to_device(device)
    }

    pub fn render(
        &mut self,
        screen: &ID3D11Texture2D,
        effect: &PaintPostprocess,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            effect.uniforms.len() == self.descriptor.uniform_size,
            "effect uniform layout changed"
        );
        let width = self.descriptor.size.width.0;
        let height = self.descriptor.size.height.0;
        anyhow::ensure!(
            effect.bounds.size.width.0 == width as f32
                && effect.bounds.size.height.0 == height as f32,
            "effect surface resized before replacement resources were ready"
        );
        let x = effect.bounds.origin.x.0;
        let y = effect.bounds.origin.y.0;
        anyhow::ensure!(
            x.is_finite() && y.is_finite() && x.fract() == 0.0 && y.fract() == 0.0,
            "effect origin must use physical pixel boundaries"
        );
        let mut screen_desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { screen.GetDesc(&mut screen_desc) };
        anyhow::ensure!(
            screen_desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM && screen_desc.SampleDesc.Count == 1,
            "effect input requires a single-sample BGRA surface"
        );
        let window = gpui::Bounds::new(
            gpui::point(gpui::ScaledPixels(0.0), gpui::ScaledPixels(0.0)),
            gpui::size(
                gpui::ScaledPixels(screen_desc.Width as f32),
                gpui::ScaledPixels(screen_desc.Height as f32),
            ),
        );
        let visible = effect.bounds.intersect(&window).intersect(&effect.content_mask.bounds);
        if visible.is_empty() || !self.input.ready_for_effect()? {
            return Ok(());
        }
        let left = visible.left().0.ceil().max(0.0) as u32;
        let top = visible.top().0.ceil().max(0.0) as u32;
        let right = visible.right().0.floor().min(screen_desc.Width as f32) as u32;
        let bottom = visible.bottom().0.floor().min(screen_desc.Height as f32) as u32;
        if left >= right || top >= bottom {
            return Ok(());
        }
        let input_box = D3D11_BOX { left, top, right, bottom, front: 0, back: 1 };
        let offset_x = (left as f32 - x) as u32;
        let offset_y = (top as f32 - y) as u32;
        let _bindings = Bindings::capture(&self.context);
        let targets = &self.resources.targets;
        unsafe {
            self.context.VSSetShaderResources(0, Some(&[None]));
            self.context.PSSetShaderResources(0, Some(&[None]));
            self.context.OMSetRenderTargets(None, None);
            self.context.ClearRenderTargetView(&targets[0].target, &[0.0; 4]);
            self.context.CopySubresourceRegion(
                &targets[0].texture,
                0,
                offset_x,
                offset_y,
                0,
                screen,
                0,
                Some(&input_box),
            );
            self.context.UpdateSubresource(
                &self.resources.uniforms,
                0,
                None,
                effect.uniforms.as_ptr().cast(),
                0,
                0,
            );
            self.context.PSSetConstantBuffers(0, Some(&[Some(self.resources.uniforms.clone())]));
            self.context.PSSetSamplers(0, Some(&[Some(self.resources.sampler.clone())]));
            self.context.OMSetBlendState(None, None, u32::MAX);
            self.context.RSSetState(None);
            self.context.RSSetViewports(Some(&[D3D11_VIEWPORT {
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
                ..Default::default()
            }]));
            self.context.IASetInputLayout(None);
            self.context.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            for (index, program) in self.resources.programs.iter().enumerate() {
                let source = &targets[index % 2];
                let output = &targets[(index + 1) % 2];
                self.context.ClearRenderTargetView(&output.target, &[0.0; 4]);
                self.context.OMSetRenderTargets(Some(&[Some(output.target.clone())]), None);
                self.context.PSSetShaderResources(0, Some(&[Some(source.view.clone())]));
                self.context.VSSetShader(&program.vertex, None);
                self.context.PSSetShader(&program.fragment, None);
                self.context.Draw(3, 0);
                self.context.PSSetShaderResources(0, Some(&[None]));
                self.context.OMSetRenderTargets(None, None);
            }
            // 在最后一段完成前不写回屏幕；输入是局部纹理，设置和相邻分屏不参与采样。
            let output = &targets[self.resources.programs.len() % 2];
            let source_box = D3D11_BOX {
                left: (left as f32 - x) as u32,
                top: (top as f32 - y) as u32,
                right: (right as f32 - x) as u32,
                bottom: (bottom as f32 - y) as u32,
                front: 0,
                back: 1,
            };
            self.context.CopySubresourceRegion(
                screen,
                0,
                left,
                top,
                0,
                &output.texture,
                0,
                Some(&source_box),
            );
        }
        self.input.signal()?;
        unsafe { self.context.Flush() };
        Ok(())
    }

    pub fn retire(self) -> anyhow::Result<StreamImageCompletion> {
        let Self { mut input, resources, .. } = self;
        input.retain_for_retirement(resources);
        input.retire()
    }
}

struct Bindings {
    context: ID3D11DeviceContext,
    target: [Option<ID3D11RenderTargetView>; 1],
    depth: Option<ID3D11DepthStencilView>,
    rasterizer: Option<ID3D11RasterizerState>,
    blend: Option<ID3D11BlendState>,
    blend_factor: [f32; 4],
    sample_mask: u32,
    constants: [Option<ID3D11Buffer>; 1],
    pixel_input: [Option<ID3D11ShaderResourceView>; 1],
    vertex_input: [Option<ID3D11ShaderResourceView>; 1],
    sampler: [Option<ID3D11SamplerState>; 1],
    pixel: Option<ID3D11PixelShader>,
    vertex: Option<ID3D11VertexShader>,
    layout: Option<ID3D11InputLayout>,
    topology: windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY,
    viewport_count: u32,
    viewports: [D3D11_VIEWPORT; 16],
}

impl Bindings {
    fn capture(context: &ID3D11DeviceContext) -> Self {
        let mut saved = Self {
            context: context.clone(),
            target: [None],
            depth: None,
            rasterizer: None,
            blend: None,
            blend_factor: [0.0; 4],
            sample_mask: 0,
            constants: [None],
            pixel_input: [None],
            vertex_input: [None],
            sampler: [None],
            pixel: None,
            vertex: None,
            layout: None,
            topology: D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            viewport_count: 16,
            viewports: [D3D11_VIEWPORT::default(); 16],
        };
        unsafe {
            context.OMGetRenderTargets(Some(&mut saved.target), Some(&mut saved.depth));
            context.OMGetBlendState(
                Some(&mut saved.blend),
                Some(&mut saved.blend_factor),
                Some(&mut saved.sample_mask),
            );
            context.PSGetConstantBuffers(0, Some(&mut saved.constants));
            context.PSGetShaderResources(0, Some(&mut saved.pixel_input));
            context.VSGetShaderResources(0, Some(&mut saved.vertex_input));
            context.PSGetSamplers(0, Some(&mut saved.sampler));
            context.PSGetShader(&mut saved.pixel, None, None);
            context.VSGetShader(&mut saved.vertex, None, None);
            // 这两个原生查询返回 void；空 COM 指针表示默认状态，而不是查询失败。
            saved.layout = context.IAGetInputLayout().ok();
            saved.rasterizer = context.RSGetState().ok();
            saved.topology = context.IAGetPrimitiveTopology();
            context.RSGetViewports(&mut saved.viewport_count, Some(saved.viewports.as_mut_ptr()));
        }
        saved
    }
}

impl Drop for Bindings {
    fn drop(&mut self) {
        unsafe {
            self.context.PSSetShaderResources(0, Some(&[None]));
            self.context.OMSetRenderTargets(Some(&self.target), self.depth.as_ref());
            self.context.PSSetConstantBuffers(0, Some(&self.constants));
            self.context.PSSetShaderResources(0, Some(&self.pixel_input));
            self.context.VSSetShaderResources(0, Some(&self.vertex_input));
            self.context.PSSetSamplers(0, Some(&self.sampler));
            self.context.PSSetShader(self.pixel.as_ref(), None);
            self.context.VSSetShader(self.vertex.as_ref(), None);
            self.context.IASetInputLayout(self.layout.as_ref());
            self.context.IASetPrimitiveTopology(self.topology);
            self.context.OMSetBlendState(
                self.blend.as_ref(),
                Some(&self.blend_factor),
                self.sample_mask,
            );
            self.context.RSSetState(self.rasterizer.as_ref());
            self.context
                .RSSetViewports(Some(&self.viewports[..self.viewport_count.min(16) as usize]));
        }
    }
}
