import os
import io
import re
import base64
import tempfile
import shutil
import sympy as sp
from sympy.parsing.latex import parse_latex
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from models.ocr_model.model.TexTeller import TexTeller
from models.ocr_model.utils.inference import inference as latex_recognition
from models.ocr_model.utils.to_katex import to_katex

st.set_page_config(
    page_title="Handwriting Math Recognition",
    page_icon="✍️"
)

# 显示标题
st.markdown("""
    <h1 style="text-align: center;">🖋 Handwriting Math Recognition</h1>
    <p style="text-align: center; font-size: 18px;">Draw your math formula below and get LaTeX output!</p>
""", unsafe_allow_html=True)

# 获取 TexTeller 模型
@st.cache_resource
def get_texteller():
    return TexTeller.from_pretrained(os.environ['CHECKPOINT_DIR'], use_onnx=False)

@st.cache_resource
def get_tokenizer():
    return TexTeller.get_tokenizer(os.environ['TOKENIZER_DIR'])

texteller = get_texteller()
tokenizer = get_tokenizer()

# 创建手写画布
st.markdown("### ✏️ Draw your math formula here:")
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # 背景透明
    stroke_width=4,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=200,
    width=600,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("🖌 Recognize Formula"):
    if canvas_result.image_data is not None:
        with st.spinner("Processing..."):
            # 将手写画布转换为 PIL 图像
            img = Image.fromarray(canvas_result.image_data.astype("uint8"))
            
            # 存储临时图片
            temp_dir = tempfile.mkdtemp()
            img_path = os.path.join(temp_dir, "handwritten_formula.png")
            img.save(img_path, "PNG")

            # 进行公式识别
            latex_result = latex_recognition(
                texteller,
                tokenizer,
                [img_path],
                accelerator="cpu",  # 你可以修改为 "cuda" 以使用 GPU
                num_beams=3
            )[0]


      # 转换为 KaTeX 可用格式
            katex_res = to_katex(latex_result)
            print(katex_res)
            latex_prs = parse_latex(katex_res)
            firstline,secondline="",""
            if "\int" in str(katex_res):
                C=""
                if str(latex_prs.args[1]).count(",")==1:
                    C="+ C"
                symbolic_result=str(sp.latex(sp.integrate(latex_prs.args[0],latex_prs.args[1])))
                firstline = katex_res+"="+symbolic_result+C
                secondline=katex_res+"="+str(sp.latex(symbolic_result))+C
            elif "\lim" in str(katex_res):
                symbolic_result = sp.limit(latex_prs.args[0] ,latex_prs.args[1], latex_prs.args[2])
                firstline=katex_res + "=" + str(sp.latex(symbolic_result))
                secondline=katex_res + "=" + str(sp.latex(symbolic_result))
            elif isinstance(latex_prs, sp.Derivative ):
                print("Derivative")
                firstline=katex_res + "=" + str(sp.latex(sp.diff(latex_prs.args[0],latex_prs.args[1])))
                secondline=katex_res + "=" + str(sp.diff(latex_prs.args[0],latex_prs.args[1]))
            elif "\sum" in str(katex_res):
                symbolic_result = sp.summation(latex_prs.args[0], (latex_prs.args[1], latex_prs.args[2], latex_prs.args[3]))
                firstline=katex_res + "=" + str(sp.latex(symbolic_result))
                secondline=katex_res + "=" + str(symbolic_result)
            
            firstline=str(firstline).replace("log","ln")
            secondline=str(secondline).replace("log","ln")
            st.success("✅ Recognition Completed!")
            st.latex(firstline)
            st.text_area("📝 LaTeX Output", secondline, height=100)
            print(type(latex_prs))
            shutil.rmtree(temp_dir)  # 清理临时文件
    else:
        st.warning("❗ Please draw something before clicking recognize.")

