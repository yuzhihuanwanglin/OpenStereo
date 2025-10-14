import onnx
import numpy as np

def inspect_quantized_weights(model_path):
    model = onnx.load(model_path)

    # 遍历所有 initializer (权重)
    for tensor in model.graph.initializer:
        if "Conv" in tensor.name:
            print(f"权重: {tensor.name}, dtype={onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor.data_type]}, shape={tensor.dims}")

    # 遍历所有节点，检查 QuantizeLinear
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            print("\n[QuantizeLinear 节点]")
            print("输入:", node.input)
            print("输出:", node.output)

            # scale
            for init in model.graph.initializer:
                if init.name == node.input[1]:  # scale
                    scale = np.frombuffer(init.raw_data, dtype=np.float32)
                    print(f"  scale: {scale}")

                if init.name == node.input[2]:  # zero_point
                    zp_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[init.data_type]
                    zp = np.frombuffer(init.raw_data, dtype=zp_dtype)
                    print(f"  zero_point: {zp} (dtype={zp_dtype})")

if __name__ == "__main__":
    inspect_quantized_weights("./output/quant/model_int8.onnx")
