import base64;
html = """<!DOCTYPE html>
<html>
<head>
	<meta charset="UTF-8">
	<title>Detection WASM Test</title>
	<style>
		body {
			margin: 20px;
			overflow-x: hidden;
		}
		#canvas {
			border: 1px solid #ccc;
			margin: 10px 0;
			max-width: 100%;
			height: auto;
			display: block;
		}
		.info {
			margin: 10px 0;
			color: #666;
		}
	</style>
</head>
<body>
	<h2>Object Detection (WASM)</h2>
	<div class="info">Upload an image for object detection</div>
	<input type="file" id="imageInput" accept="image/*">
	<canvas id="canvas"></canvas>
	<div id="result"></div>
	<script>
		const canvas = document.getElementById('canvas');
		const ctx = canvas.getContext('2d', { willReadFrequently: true });
		WebAssembly.instantiateStreaming(fetch(""))
			.then(({ instance }) => {
				const wasm = instance.exports;
				wasm.wasm_init();
				const imageBufferOffset = wasm.wasm_image_buffer();
				const wasmMemory = new Uint8Array(wasm.memory.buffer);
				document.getElementById('imageInput').addEventListener('change', (e) => {
					const file = e.target.files[0];
					if (!file) return;
					const img = new Image();
					img.onload = () => {
						const w = img.width, h = img.height;
						canvas.width = w;
						canvas.height = h;
						ctx.drawImage(img, 0, 0);
						const imageData = ctx.getImageData(0, 0, w, h);
						const src = imageData.data;
						for (let i = 0, j = imageBufferOffset; i < src.length; i += 4, j += 3) {
							wasmMemory[j] = src[i];		 
							wasmMemory[j + 1] = src[i + 1]; 
							wasmMemory[j + 2] = src[i + 2]; 
						}
						var start_time = performance.now();
						const numDetections = wasm.wasm_detect(w, h);
						document.getElementById('result').textContent = `Detected ${numDetections} objects, time: ${((performance.now() - start_time) / 1000).toFixed(4)}s`;
						const resultData = ctx.createImageData(w, h);
						const dst = resultData.data;
						for (let i = 0, j = imageBufferOffset; i < dst.length; i += 4, j += 3) {
							dst[i] = wasmMemory[j];
							dst[i + 1] = wasmMemory[j + 1];
							dst[i + 2] = wasmMemory[j + 2];
							dst[i + 3] = 255;
						}
						ctx.putImageData(resultData, 0, 0);
					};
					img.src = URL.createObjectURL(file);
				});
			})
			.catch(err => console.error('WASM load failed:', err));
	</script>
</body>
</html>
"""
path = "mb_tiny.wasm"
with open(path, 'rb') as f:
	data = f.read()
	b64_data = base64.b64encode(data).decode('ascii')
	html_with_wasm = html.replace('fetch("")', f'fetch("data:application/wasm;base64,{b64_data}")')
	with open(f"{path}.html", 'w') as w:
		w.write(html_with_wasm)