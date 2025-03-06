# import os
# import io
# import re
# import base64
# import tempfile
# import shutil
# import streamlit as st

# from PIL import Image
# from streamlit_paste_button import paste_image_button as pbutton
# from onnxruntime import InferenceSession
# from models.thrid_party.paddleocr.infer import predict_det, predict_rec
# from models.thrid_party.paddleocr.infer import utility

# from models.utils import mix_inference
# from models.det_model.inference import PredictConfig

# from models.ocr_model.model.TexTeller import TexTeller
# from models.ocr_model.utils.inference import inference as latex_recognition
# from models.ocr_model.utils.to_katex import to_katex


# st.set_page_config(
#     page_title="TexTeller",
#     page_icon="🧮"
# )

# html_string = '''
#     <h1 style="color: black; text-align: center;">
#         <img src="https://raw.githubusercontent.com/OleehyO/TexTeller/main/assets/fire.svg" width="100">
#         𝚃𝚎𝚡𝚃𝚎𝚕𝚕𝚎𝚛
#         <img src="https://raw.githubusercontent.com/OleehyO/TexTeller/main/assets/fire.svg" width="100">
#     </h1>
# '''

# suc_gif_html = '''
#     <h1 style="color: black; text-align: center;">
#         <img src="https://slackmojis.com/emojis/90621-clapclap-e/download" width="50">
#         <img src="https://slackmojis.com/emojis/90621-clapclap-e/download" width="50">
#         <img src="https://slackmojis.com/emojis/90621-clapclap-e/download" width="50">
#     </h1>
# '''

# fail_gif_html = '''
#     <h1 style="color: black; text-align: center;">
#         <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download" >
#         <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download" >
#         <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download" >
#     </h1>
# '''

# @st.cache_resource
# def get_texteller(use_onnx, accelerator):
#     return TexTeller.from_pretrained(os.environ['CHECKPOINT_DIR'], use_onnx=use_onnx, onnx_provider=accelerator)

# @st.cache_resource
# def get_tokenizer():
#     return TexTeller.get_tokenizer(os.environ['TOKENIZER_DIR'])

# @st.cache_resource
# def get_det_models(accelerator):
#     infer_config = PredictConfig("./models/det_model/model/infer_cfg.yml")
#     latex_det_model = InferenceSession(
#         "./models/det_model/model/rtdetr_r50vd_6x_coco.onnx", 
#         providers=['CUDAExecutionProvider'] if accelerator == 'cuda' else ['CPUExecutionProvider']
#     )
#     return infer_config, latex_det_model

# @st.cache_resource()
# def get_ocr_models(accelerator):
#     use_gpu = accelerator == 'cuda'

#     SIZE_LIMIT = 20 * 1024 * 1024
#     det_model_dir = "./models/thrid_party/paddleocr/checkpoints/det/default_model.onnx"
#     rec_model_dir = "./models/thrid_party/paddleocr/checkpoints/rec/default_model.onnx"
#     # The CPU inference of the detection model will be faster than the GPU inference (in onnxruntime)
#     det_use_gpu = False
#     rec_use_gpu = use_gpu and not (os.path.getsize(rec_model_dir) < SIZE_LIMIT)

#     paddleocr_args = utility.parse_args()
#     paddleocr_args.use_onnx = True
#     paddleocr_args.det_model_dir = det_model_dir
#     paddleocr_args.rec_model_dir = rec_model_dir

#     paddleocr_args.use_gpu = det_use_gpu
#     detector = predict_det.TextDetector(paddleocr_args)
#     paddleocr_args.use_gpu = rec_use_gpu
#     recognizer = predict_rec.TextRecognizer(paddleocr_args)
#     return [detector, recognizer]


# def get_image_base64(img_file):
#     buffered = io.BytesIO()
#     img_file.seek(0)
#     img = Image.open(img_file)
#     img.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode()

# def on_file_upload():
#     st.session_state["UPLOADED_FILE_CHANGED"] = True

# def change_side_bar():
#     st.session_state["CHANGE_SIDEBAR_FLAG"] = True

# if "start" not in st.session_state:
#     st.session_state["start"] = 1
#     st.toast('Hooray!', icon='🎉')

# if "UPLOADED_FILE_CHANGED" not in st.session_state:
#     st.session_state["UPLOADED_FILE_CHANGED"] = False

# if "CHANGE_SIDEBAR_FLAG" not in st.session_state:
#     st.session_state["CHANGE_SIDEBAR_FLAG"] = False

# if "INF_MODE" not in st.session_state:
#     st.session_state["INF_MODE"] = "Formula recognition"


# ##############################     <sidebar>    ##############################

# with st.sidebar:
#     num_beams = 1

#     st.markdown("# 🔨️ Config")
#     st.markdown("")

#     inf_mode = st.selectbox(
#         "Inference mode",
#         ("Formula recognition", "Paragraph recognition"),
#         on_change=change_side_bar
#     )

#     num_beams = st.number_input(
#         'Number of beams',
#         min_value=1,
#         max_value=20,
#         step=1,
#         on_change=change_side_bar
#     )

#     accelerator = st.radio(
#         "Accelerator",
#         ("cpu", "cuda", "mps"),
#         on_change=change_side_bar
#     )

#     st.markdown("## Seedup")
#     use_onnx = st.toggle("ONNX Runtime ")



# ##############################     </sidebar>    ##############################


# ################################     <page>    ################################

# texteller = get_texteller(use_onnx, accelerator)
# tokenizer = get_tokenizer()
# latex_rec_models = [texteller, tokenizer]

# if inf_mode == "Paragraph recognition":
#     infer_config, latex_det_model = get_det_models(accelerator)
#     lang_ocr_models = get_ocr_models(accelerator)

# st.markdown(html_string, unsafe_allow_html=True)

# uploaded_file = st.file_uploader(
#     " ",
#     type=['jpg', 'png'],
#     on_change=on_file_upload
# )

# paste_result = pbutton(
#     label="📋 Paste an image",
#     background_color="#5BBCFF",
#     hover_background_color="#3498db",
# )
# st.write("")

# if st.session_state["CHANGE_SIDEBAR_FLAG"] == True:
#     st.session_state["CHANGE_SIDEBAR_FLAG"] = False
# elif uploaded_file or paste_result.image_data is not None:
#     if st.session_state["UPLOADED_FILE_CHANGED"] == False and paste_result.image_data is not None:
#         uploaded_file = io.BytesIO()
#         paste_result.image_data.save(uploaded_file, format='PNG')
#         uploaded_file.seek(0)

#     if st.session_state["UPLOADED_FILE_CHANGED"] == True:
#         st.session_state["UPLOADED_FILE_CHANGED"] = False

#     img = Image.open(uploaded_file)

#     temp_dir = tempfile.mkdtemp()
#     png_file_path = os.path.join(temp_dir, 'image.png')
#     img.save(png_file_path, 'PNG')

#     with st.container(height=300):
#         img_base64 = get_image_base64(uploaded_file)

#         st.markdown(f"""
#         <style>
#         .centered-container {{
#             text-align: center;
#         }}
#         .centered-image {{
#             display: block;
#             margin-left: auto;
#             margin-right: auto;
#             max-height: 350px;
#             max-width: 100%;
#         }}
#         </style>
#         <div class="centered-container">
#             <img src="data:image/png;base64,{img_base64}" class="centered-image" alt="Input image">
#         </div>
#         """, unsafe_allow_html=True)
#     st.markdown(f"""
#     <style>
#     .centered-container {{
#         text-align: center;
#     }}
#     </style>
#     <div class="centered-container">
#         <p style="color:gray;">Input image ({img.height}✖️{img.width})</p>
#     </div>
#     """, unsafe_allow_html=True)

#     st.write("")

#     with st.spinner("Predicting..."):
#         if inf_mode == "Formula recognition":
#             TexTeller_result = latex_recognition(
#                 texteller,
#                 tokenizer,
#                 [png_file_path],
#                 accelerator=accelerator,
#                 num_beams=num_beams
#             )[0]
#             katex_res = to_katex(TexTeller_result)
#         else:
#             katex_res = mix_inference(png_file_path, infer_config, latex_det_model, lang_ocr_models, latex_rec_models, accelerator, num_beams)

#         st.success('Completed!', icon="✅")
#         st.markdown(suc_gif_html, unsafe_allow_html=True)
#         st.text_area(":blue[***  𝑃r𝑒d𝑖c𝑡e𝑑 𝑓o𝑟m𝑢l𝑎  ***]", katex_res, height=150)

#         if inf_mode == "Formula recognition":
#             st.latex(katex_res)
#         elif inf_mode == "Paragraph recognition":
#             mixed_res = re.split(r'(\$\$.*?\$\$)', katex_res)
#             for text in mixed_res:
#                 if text.startswith('$$') and text.endswith('$$'):
#                     st.latex(text[2:-2])
#                 else:
#                     st.markdown(text)

#         st.write("")
#         st.write("")

#         with st.expander(":star2: :gray[Tips for better results]"):
#             st.markdown('''
#                 * :mag_right: Use a clear and high-resolution image.
#                 * :scissors: Crop images as accurately as possible.
#                 * :jigsaw: Split large multi line formulas into smaller ones.
#                 * :page_facing_up: Use images with **white background and black text** as much as possible.
#                 * :book: Use a font with good readability.
#             ''')
#         shutil.rmtree(temp_dir)

#     paste_result.image_data = None

# ################################     </page>    ################################


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
    page_title="TexTeller Handwriting",
    page_icon="✍️"
)

# 显示标题
st.markdown("""
    <h1 style="text-align: center;">🖋 TexTeller - Handwriting Math Recognition</h1>
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
                accelerator="mps",  # 你可以修改为 "cuda" 以使用 GPU
                num_beams=3
            )[0]

            # # 转换为 KaTeX 可用格式
            # katex_res = to_katex(latex_result)
            # print(katex_res)
            # latex_prs = parse_latex(katex_res)
            # firstline,secondline="",""
            # if "\int" in str(katex_res):
            #     symbolic_result = sp.integrate(latex_prs.args[0],latex_prs.args[1])
            #     print(str(latex_prs.args[1]))
            #     st.success("✅ Recognition Completed!")
            #     C=""
            #     if str(latex_prs.args[1]).count(",")==1:
            #         C="+ C"
            #     st.latex(katex_res+"="+str(sp.latex(symbolic_result)).replace("log","ln")+C)
            #     st.text_area("📝 LaTeX Output", katex_res+"="+str(sp.latex(symbolic_result)).replace("log","ln")+C, height=100)
            # elif "\lim" in str(katex_res):
            #     st.success("✅ Recognition Completed!")
            #     limit_function = latex_prs.args[0]
            #     limit_variable = latex_prs.args[1]
            #     limit_point = latex_prs.args[2]
            #     result = sp.limit(limit_function, limit_variable, limit_point)
            #     st.latex(katex_res + "=" + str(sp.latex(result)).replace("log","ln"))
            #     st.text_area("📝 LaTeX Output", katex_res + "=" + str(sp.latex(result)).replace("log","ln"), height=100)
            # elif isinstance(latex_prs, sp.Derivative ):
            #     print("Derivative")
            #     st.success("✅ Recognition Completed!")
            #     st.latex(katex_res + "=" + str(sp.latex(sp.diff(latex_prs.args[0],latex_prs.args[1]))))
            #     st.text_area("📝 LaTeX Output", katex_res + "=" + str(sp.diff(latex_prs.args[0],latex_prs.args[1])), height=100)
            #     print(sp.diff(latex_prs.args[0],latex_prs.args[1]))
            # print(type(latex_prs))

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
            firstline=str(firstline).replace("log","ln")
            secondline=str(secondline).replace("log","ln")
            st.success("✅ Recognition Completed!")
            st.latex(firstline)
            st.text_area("📝 LaTeX Output", secondline, height=100)
            print(type(latex_prs))
            shutil.rmtree(temp_dir)  # 清理临时文件
    else:
        st.warning("❗ Please draw something before clicking recognize.")

