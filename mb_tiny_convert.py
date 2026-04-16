import mb_tiny, torch, os
state_dict_path = "mb_tiny.pth"
model = mb_tiny.Model().cpu()
model.load_pretrained_weights(state_dict_path)
model.float()
state_dict = model.state_dict()
file, ext = os.path.splitext(os.path.basename(state_dict_path))
offset = 0
with open(f"{state_dict_path}.c", 'w') as c:
	lens={}
	c.write("const static float {}_data[] = {{".format(file))
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Conv2d):
			weight = module.weight.detach().flatten().numpy()
			bias = module.bias.detach().flatten().numpy()
			lens[name] = (len(weight), len(bias))
			c_array = weight.tolist() + bias.tolist()
			c_str = ", ".join([f"{v:.8f}f" for v in c_array])
			c.write(f"{c_str}, ")
	c.write("};\nstruct convLayer{\n	const float* weights;\n	const float* bias;\n	unsigned in_channels;\n	unsigned out_channels;\n	unsigned kernel_size;\n	unsigned stride;\n	unsigned padding;\n	unsigned groups;\n};\n")
	c.write(f"const struct convLayer {file}[{len(lens)}] = {{")
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Conv2d):
			weight_len, bias_len = lens[name]
			c.write("\t(struct convLayer){"
				f".weights = {file}_data + {offset}, "
				f".bias = {file}_data + {offset + weight_len}, "
				f".in_channels = {module.in_channels}, "
				f".out_channels = {module.out_channels}, "
				f".kernel_size = {module.kernel_size[0]}, "
				f".stride = {module.stride[0]}, "
				f".padding = {module.padding[0]}, "
				f".groups = {module.groups}, "
				"}, // "
				f"{name}\n"
			)
			offset += weight_len + bias_len
	c.write("};\n")

# 单独循环打印每层使用的优化函数
layer_idx = 0
# 跟踪缓冲区使用情况和输入尺寸
buf_in = "buf_a"
buf_out = "buf_b"
in_h, in_w = 240, 320  # 初始输入尺寸
in_ch = 3  # 初始输入通道
# 跟踪 buf_a 和 buf_b 的最大使用量
buf_a_max = 0
buf_b_max = 0
for name, module in model.named_modules():
	if isinstance(module, torch.nn.Conv2d):
		k = module.kernel_size[0]
		s = module.stride[0]
		p = module.padding[0]
		g = module.groups
		in_ch = module.in_channels
		out_ch = module.out_channels
		
		# 计算输出尺寸
		out_h = (in_h + 2 * p - k) // s + 1
		out_w = (in_w + 2 * p - k) // s + 1
		
		# 判断逻辑（基于mb_tiny.c中的实际使用）
		# detection_head_fused 用于检测头的 DW+PW 融合
		# DW 层: 15, 17, 25, 27, 33, 35
		# PW 层: 16, 18, 26, 28, 34, 36 (与DW层配对使用，跳过不打印)
		detection_head_dw_layers = {15, 17, 25, 27, 33, 35}
		detection_head_pw_layers = {16, 18, 26, 28, 34, 36}
		
		# 更新缓冲区最大值
		def update_buf_max(buf_in_name, in_size, out_size):
			global buf_a_max, buf_b_max
			if buf_in_name == "buf_a":
				buf_a_max = max(buf_a_max, in_size)
				buf_b_max = max(buf_b_max, out_size)
			else:
				buf_b_max = max(buf_b_max, in_size)
				buf_a_max = max(buf_a_max, out_size)
		
		if layer_idx in detection_head_pw_layers:
			# 跳过被融合的 PW 层，但更新尺寸
			in_ch = out_ch
			in_h, in_w = out_h, out_w
			layer_idx += 1
			continue
		elif layer_idx in detection_head_dw_layers:
			pw_idx = layer_idx + 1
			# 获取所有 Conv2d 层的列表
			conv_modules = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]
			pw_module = conv_modules[pw_idx][1] if pw_idx < len(conv_modules) else None
			pw_out_ch = pw_module.out_channels if pw_module else out_ch
			out_size = out_h * out_w * pw_out_ch
			func_call = f"detection_head_fused({file}[{layer_idx}], {file}[{pw_idx}], {buf_in}, {in_h}, {in_w}, out[{out_size}], {buf_out})"
			update_buf_max(buf_in, in_ch * in_h * in_w, out_size)
			buf_in, buf_out = buf_out, buf_in
			in_ch = pw_out_ch
			in_h, in_w = out_h, out_w
		elif k == 1 and s == 1 and p == 0 and g == 1:
			func_call = f"conv1x1_relu_forward({file}[{layer_idx}], {buf_in}, {in_h}, {in_w}, {buf_out})"
			update_buf_max(buf_in, in_ch * in_h * in_w, in_ch * in_h * in_w)
			buf_in, buf_out = buf_out, buf_in
		elif layer_idx in [40, 41]:
			out_size = out_h * out_w * out_ch
			func_call = f"conv2d_forward_permute_reshape({file}[{layer_idx}], {buf_in}, {in_h}, {in_w}, out[{out_size}])"
		else:
			func_call = f"conv2d_forward_relu({file}[{layer_idx}], {buf_in}, {in_h}, {in_w}, {buf_out})"
			update_buf_max(buf_in, in_ch * in_h * in_w, out_ch * out_h * out_w)
			buf_in, buf_out = buf_out, buf_in
			in_ch = out_ch
			in_h, in_w = out_h, out_w
		
		print(f"{func_call} // {name}")
		layer_idx += 1

print(f"float buf_a[{buf_a_max}]")
print(f"float buf_b[{buf_b_max}]")