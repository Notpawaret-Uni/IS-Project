import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
import sklearn
from pathlib import Path
from textwrap import dedent 

# Resolve files relative to this script so Streamlit Cloud/project root differences
# do not cause FileNotFoundError when the app is started from the repository root.
BASE_DIR = Path(__file__).resolve().parent

# หน้าเว็บ: ตั้งค่าหน้าตา
st.set_page_config(page_title='F1 Predictor', page_icon='🏎️', layout='wide')

_CSS = '''
<style>
/* Background and card styles */
body { background-color: #0f1724; }
.stApp { color-scheme: dark; }
.big-title {
    font-size: 48px;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -1px;
    line-height: 1.05;
    margin-bottom: 6px;
    text-transform: uppercase;
    text-shadow: 0 0 14px rgba(248,250,252,0.08);
}
.sub {
    font-size: 20px;
    color: #94A3B8;
    font-weight: 400;
    margin-bottom: 18px;
}

/* Markdown / write text styling */
.stMarkdown, .stMarkdown p, .stText, .stWrite {
    color: #E6EEF8;
    font-size: 18px;
    line-height: 1.6;
    font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #F8FAFC;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.section {
    background: linear-gradient(90deg, rgba(20,20,30,0.6), rgba(10,10,20,0.6));
    padding: 12px;
    border-radius: 8px;
}
.result {
    background: linear-gradient(90deg, rgba(6,95,70,0.15), rgba(6,50,95,0.05));
    padding: 14px;
    border-radius: 8px;
}
.footer { color: #94A3B8; font-size: 12px; }
/* Buttons contrast */
.stButton>button { background-color: #0b1220; color: #E6EEF8; border-radius: 8px; }
</style>
'''


st.markdown(_CSS, unsafe_allow_html=True)

# --- 1. โหลดข้อมูลและสร้างระบบ Mapping ---
@st.cache_resource
def setup_environment():
    df_real_names = pd.read_csv(str(BASE_DIR / 'Formula1_Pitstop_Data_1950-2024_all_rounds.csv'))
    df_train = pd.read_csv(str(BASE_DIR / 'f1_enhanced_dataset_for_analysis.csv'))
    loaded = None
    try:
        loaded = joblib.load(str(BASE_DIR / 'f1_ensemble_model_v3.pkl'))
    except Exception:
        loaded = None

    nn = None
    try:
        nn = load_model(str(BASE_DIR / 'f1_nn_model_v3.h5'))
    except Exception:
        nn = None

    # Try to detect if the loaded model is a VotingClassifier (ensemble of RF/GB/LR)
    voting_model = None
    rf_model = None
    try:
        # If it's a VotingClassifier-like object, it may have named_estimators_
        if hasattr(loaded, 'named_estimators'):
            voting_model = loaded
            named = getattr(loaded, 'named_estimators')
            # try common keys
            for k, est in named.items():
                if isinstance(est, RandomForestClassifier) or 'forest' in k.lower() or 'rf' in k.lower():
                    rf_model = est
                    break
            # fallback: search values
            if rf_model is None:
                for est in named.values():
                    if isinstance(est, RandomForestClassifier):
                        rf_model = est
                        break
        elif hasattr(loaded, 'estimators_'):
            # older style: estimators_ list
            voting_model = loaded
            for est in getattr(loaded, 'estimators_'):
                if isinstance(est, RandomForestClassifier):
                    rf_model = est
                    break
        else:
            # not an ensemble: treat loaded as a single RF model
            rf_model = loaded
    except Exception:
        # on any failure, fallback: assume loaded is RF
        rf_model = loaded
        voting_model = None

    # If still no rf_model found, try to assign loaded itself
    if rf_model is None:
        rf_model = loaded

    seasons = sorted(df_real_names['Season'].unique())
    circuits_by_season = {}
    drivers_by_season = {}
    teams_by_season = {}
    teams_by_season_and_driver = {}
    drivers_by_season_and_team = {}

    for s in seasons:
        df_s = df_real_names[df_real_names['Season'] == s]
        circuits = sorted(df_s['Circuit'].dropna().unique())
        drivers = sorted(df_s['Driver'].dropna().unique())
        teams = sorted(df_s['Constructor'].dropna().unique())
        circuits_by_season[s] = circuits
        drivers_by_season[s] = drivers
        teams_by_season[s] = teams

        # map driver->teams in that season (drivers may have multiple constructors)
        for d in drivers:
            teams_for_d = sorted(df_s[df_s['Driver'] == d]['Constructor'].dropna().unique())
            teams_by_season_and_driver[(s, d)] = teams_for_d

        # map team->drivers in that season
        for t in teams:
            drivers_for_t = sorted(df_s[df_s['Constructor'] == t]['Driver'].dropna().unique())
            drivers_by_season_and_team[(s, t)] = drivers_for_t

    mapping = {
        'seasons': seasons,
        'circuits_by_season': circuits_by_season,
        'drivers_by_season': drivers_by_season,
        'teams_by_season': teams_by_season,
        'teams_by_season_and_driver': teams_by_season_and_driver,
        'drivers_by_season_and_team': drivers_by_season_and_team,
        'global_drivers': sorted(df_real_names['Driver'].dropna().unique()),
        'global_teams': sorted(df_real_names['Constructor'].dropna().unique()),
        'global_gps': sorted(df_real_names['Circuit'].dropna().unique()),
        'weather': ["Sunny", "Rainy", "Cloudy", "Windy"],
        'tires': ["Soft-Hard", "Medium-Hard", "Soft-Medium", "Two-stop"]
    }
    return rf_model, voting_model, nn, mapping


rf_model, voting_model, nn_model, m = setup_environment()

def encode_selection(selection, categories_list):
    index = categories_list.index(selection)
    return index / (len(categories_list) - 1) if len(categories_list) > 1 else 0.0

# Header
st.markdown('<div class="big-title">F1 TOP 10 Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">ระบบวิเคราะห์และทำนายอันดับการเข้าเส้นชัยใน 10 อันดับแรกของการแข่งขัน Formula 1 </div>', unsafe_allow_html=True)
st.write('')

# (Moved analysis UI into the Test tabs below; removed sidebar inputs.)

# ====================== NAVIGATION BAR ด้านบน ======================
st.markdown("---")  # เส้นแบ่งให้ดูเป็น nav bar ชัด

tabs = st.tabs([
    "Dataset Features", 
    "อธิบาย MACHINE LEARNING ENSEMBLE", 
    "อธิบาย NEURAL NETWORK",
    "ทดสอบ Model 1 (Ensemble)", 
    "ทดสอบ Model 2 (Neural Network)"
])

# ====================== TAB 1: Dataset Features ======================
with tabs[0]:
    
    st.success("### ที่มาและความสำคัญ")
    st.write(""" 
        โปรเจกต์นี้มีจุดประสงค์เพื่อวิเคราะห์และทำนายอันดับการเข้าเส้นชัยใน 10 อันดับแรกของการแข่งขัน Formula 1 โดยใช้เทคนิค Machine Learning และ Neural Network โดยใช้ข้อมูลตั้งแต่ปี 1950 ถึง 2024 ซึ่งเป็นยุคที่มีการเปลี่ยนแปลงอย่างมากในเทคโนโลยีรถแข่งและกลยุทธ์การแข่ง การวิเคราะห์นี้จะช่วยให้เข้าใจปัจจัยที่มีผลต่อความสำเร็จของนักแข่งและทีมในแต่ละฤดูกาล และสามารถทำนายผลการแข่งขันได้อย่างแม่นยำมากขึ้น
    """)
    
    st.success("### อธิบาย Feature ของ Dataset")
    
    st.markdown("### Dataset 1: Formula1_Pitstop_Data_1950-2024_all_rounds.csv")
    st.write(""" ชุดข้อมูลนี้เน้นไปที่ข้อมูลเชิงประวัติศาสตร์และสถิติในพิทสตอป (Pit Stop) ของการแข่งขัน F1 ตั้งแต่ปี 1950 ถึง 2024 โดยมีฟีเจอร์หลัก ๆ เช่น:
    * **Season / Year** : ปีที่ทำการแข่งขัน (ใช้เชื่อมโยงข้อมูลข้ามชุด)
    * **Circuit** : ชื่อสนามแข่งขันหรือรายการแกรนด์ปรีซ์
    * **Driver** : ชื่อนักแข่ง (Key หลักในการระบุตัวบุคคล)
    * **Constructor** : ชื่อทีมผู้สร้างรถแข่ง
    * **Pit Stops** : จำนวนครั้งที่นักแข่งเข้าพิทในเรซนั้นๆ
    * **Average Pit Time** : เวลาเฉลี่ยที่ใช้ในหลุมพิท (วินาที)
    * **Weather Conditions** : สภาพอากาศและอุณหภูมิสนาม (มีผลต่อการเลือกยางและสมรรถนะรถ)
    """)
    
    st.markdown("### Dataset 2: f1_enhanced_dataset_for_analysis")
    st.write("""
    ชุดข้อมูลนี้เป็นข้อมูลเชิงลึกที่ใช้สำหรับการวิเคราะห์เชิงทำนาย (Predictive Analysis)

    **1. ข้อมูลก่อนเริ่มแข่ง (Input Features/Predictors)**
    นี่คือกลุ่มตัวแปรที่ใช้ป้อนให้ AI (ML และ NN) เพื่อทายผล:
    - **StartPosition (Grid)**: อันดับการสตาร์ทที่จุดปล่อยตัว (เป็น Feature ที่มีผลสูงสุดต่อผลลัพธ์)
    - **QualifyingPosition**: อันดับที่ทำได้ในรอบคัดเลือกก่อนวันแข่งจริง
    - **TireStrategy**: แผนการใช้ยาง (เช่น Soft-Hard, Medium-Hard)
    - **Weather Condition**: สภาพอากาศในวันแข่ง (Sunny, Rainy, Cloudy)

    **2. ข้อมูลหลังจบการแข่ง (Post-race Data / Data Leakage)**
    ตัวแปรกลุ่มนี้ต้องตัดออก (Drop) ตอนเทรนโมเดล เพราะเป็นสิ่งที่ "รู้ผลหลังแข่งจบ" เท่านั้น:
    - **DNF (Did Not Finish)**: สถานะการแข่งไม่จบ (เช่น รถเสีย หรืออุบัติเหตุ)
    - **Overtakes**: จำนวนครั้งที่มีการแซงเกิดขึ้นในเรซนั้น
    - **Points**: คะแนนที่ได้รับจริงหลังจบการแข่ง

    **3. ตัวแปรเป้าหมาย (Target Variable)**
    - **FinishPosition**: อันดับที่เข้าเส้นชัยจริง
    """)
    
    st.divider()
    st.success("### ขั้นตอนการ Clean Data (จาก Colab)")
    
    col_clean1, col_clean2 = st.columns(2)
    
    with col_clean1:
        st.markdown("""
        **1. Data Integration**
        - ทำการเชื่อมโยงข้อมูลจาก 2 Dataset หลักเข้าด้วยกัน คือ ข้อมูลการแข่งขัน (Race Data) และ ข้อมูลพิทสตอป (Pit Stop Data) โดยใช้คีย์หลัก Key คือ `Season` และ `Driver` เพื่อให้ได้ภาพรวมสถิติของนักแข่งแต่ละคนในแต่ละสนามอย่างครบถ้วน
        
        **2. Data Leakage Prevention**
        - ทำการคัดออก (Drop) คอลัมน์ที่จะทราบผลก็ต่อเมื่อการแข่งขันจบลงแล้วเท่านั้น เช่น Points, RaceTime_sec, DNF, และ Overtakes หากเก็บค่าเหล่านี้ไว้ โมเดลจะทายถูกเกือบ 100% แต่จะใช้งานจริงไม่ได้
        
        **3. Missing Value Handling**
        - ข้อมูลบางส่วนโดยเฉพาะสถิติพิทสตอปในอดีตอาจมีค่าว่าง (Missing Values) จึงเติมค่าว่างด้วยค่าเฉลี่ย (Mean) ของนักแข่งทั้งหมดและเติมค่าว่างด้วย 0 (สมมติว่าไม่มีการเข้าพิท)
        """)
        
    with col_clean2:
        st.markdown("""
        **4. Label Encoding**
        - ใช้เทคนิค Label Encoding เพื่อแปลงชื่อนักแข่ง (Driver), ชื่อทีม (Team), สนามแข่ง (Location), และสภาพอากาศ (Weather) ให้กลายเป็นตัวเลขลำดับ เพื่อให้โมเดลสามารถนำไปคำนวณทางคณิตศาสตร์ได้
        
        **5. MinMaxScaler**
        - ปรับข้อมูลทุกตัวให้อยู่ในช่วง $[0, 1]$ เพื่อลด Bias ของตัวเลขที่มีค่าสูง หากไม่ปรับสเกล โมเดลจะให้ความสำคัญกับตัวเลขที่มากกว่าโดยผิดพลาด การปรับให้เป็น $[0, 1]$ ช่วยให้ Neural Network เรียนรู้ได้เร็วและมีประสิทธิภาพสูงสุด
        """)
    
    st.divider()
    st.subheader("แหล่งอ้างอิงข้อมูล (References)")
    st.write("""
    1. Akash Rane. (2024). Formula 1 Pit Stop Dataset. Kaggle.  
    https://www.kaggle.com/datasets/akashrane2609/formula-1-pit-stop-dataset  

    2. Usman. (2024). Formula 1 Race Dataset for Predictive Analysis. Kaggle.  
    https://www.kaggle.com/datasets/usman136/formula-1-race-dataset-for-predictive-analysis
    """)

# ====================== TAB 2: Model 1 Explanation ======================
with tabs[1]:
    st.markdown("### Model 1: Machine Learning Ensemble")
    st.write("""
    **แนวทางการพัฒนาโมเดล Machine Learning (Ensemble)**
    """)

    st.success("### 1. การเตรียมข้อมูล (Data Preparation & Feature Engineering)")
    st.markdown("""
    * **Data Integration**: เชื่อมโยงข้อมูลจากสถิติพิทสตอป (Pit Stop) และผลการแข่งเชิงลึก โดยใช้ **Season** และ **Driver** เป็น Key หลัก
    * **Data Integrity Check**: ระบบตรวจสอบความถูกต้องเชิงประวัติศาสตร์ เพื่อตัดข้อมูล "นักแข่ง-ทีม-ปี" ที่ไม่ตรงตามความเป็นจริงออก
    * **Handling Data Leakage**: การคัดออก (Drop) ฟีเจอร์ที่ล่วงรู้ผลอนาคต เช่น *DNF, Overtakes, และเวลาพิทสตอป* เพื่อให้โมเดลทำนายจากข้อมูลก่อนเริ่มแข่ง (Pre-race) ได้จริง
    * **Feature Engineering & Scaling**: 
        * แปลงข้อมูลข้อความด้วย **Label Encoding**
        * ปรับช่วงข้อมูลเป็น [0, 1] ด้วย **MinMaxScaler** เพื่อป้องกันปัญหาตัวแปรที่มีค่าสูง (เช่น ปี ค.ศ.) ข่มตัวแปรสำคัญอื่น ๆ
    """)
    
    st.write("**MinMaxScaler**")
    st.latex(r"x' = \frac{x - \min(x)}{\max(x) - \min(x)}")
    
    st.success("### 2. ทฤษฎีอัลกอริทึม (Ensemble Theory)")
    st.markdown("""
    โมเดลนี้ใช้เทคนิค **Voting Classifier (Soft Voting)** ซึ่งเป็นการมัดรวมพลังจาก 3 อัลกอริทึมที่แตกต่างกันเพื่อลดความผิดพลาด:
    1.  **Random Forest**: ใช้การโหวตจากต้นไม้ตัดสินใจจำนวนมาก (Bagging)
    2.  **Gradient Boosting**: พัฒนาความแม่นยำผ่านการเรียนรู้จากข้อผิดพลาด (Boosting)
    3.  **Logistic Regression**: ใช้สมการความน่าจะเป็นทางคณิตศาสตร์เพื่อทำนายการแยกแยะกลุ่มข้อมูล (Classification)
    """)
    
    st.write("**Logistic Regression**")
    st.latex(r"P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x)}}")

    st.write("**การรวมผลลัพธ์ (Soft Voting):**")
    st.info("""
    ใช้วิธี **Soft Voting** ในการรวมพลังของทั้ง 3 โมเดล โดยการนำ 'ค่าความน่าจะเป็น' 
    ที่แต่ละโมเดลทำนายได้มาหาค่าเฉลี่ยถ่วงน้ำหนัก วิธีนี้ช่วยให้ผลการทายผลมีความเสถียร 
    และสะท้อนถึงความมั่นใจของ AI ได้แม่นยำยิ่งขึ้นกว่าแบบใช้โมเดลตัวเดียว
    """)
    
    st.success("### 3. ขั้นตอนการพัฒนาโมเดล (Model Development)")
    
    col_ens1, col_ens2 = st.columns(2)
    
    with col_ens1:
        st.markdown("""
        **1. การแบ่งข้อมูลสำหรับการทดสอบ (80/20)**
        - ก่อนการฝึกฝน แบ่งข้อมูลออกเป็น 2 ชุด คือ
             * `Training Set (80%)`: ใช้สำหรับให้โมเดลเรียนรู้ความสัมพันธ์ของข้อมูล
             * `Test Set (20%)`: ใช้สำหรับวัดผลความแม่นยำด้วยข้อมูลที่โมเดลไม่เคยเห็นมาก่อน (Unseen Data) เพื่อยืนยันว่าโมเดลใช้งานได้จริง
        
        **2. การเลือกอัลกอริทึมพื้นฐาน**
        - เลือกใช้โมเดล 3 ประเภทที่มีจุดเด่นต่างกัน เพื่อให้นำมาเสริมจุดแข็งและอุดจุดอ่อนซึ่งกันและกัน
             * `Random Forest (RF)`: ใช้การสร้างต้นไม้ตัดสินใจหลาย ๆ ต้นแล้วหาเสียงส่วนมาก ช่วยลดความแปรปรวน (Variance)
             * `Gradient Boosting (GB)`: เรียนรู้จากข้อผิดพลาดของรอบก่อนหน้าเพื่อเพิ่มความแม่นยำ (Bias Reduction)
             * `Logistic Regression (LR)`: ใช้สมการความน่าจะเป็นเชิงเส้นเพื่อหาเกณฑ์ในการจำแนกที่ชัดเจน
    
        """)
        
    with col_ens2:
        st.markdown("**3. Soft Voting Strategy**")
        
        st.latex(r"\hat{P}(y|x) = \frac{1}{n} \sum_{i=1}^{n} P_i(y|x)")
        
        st.markdown("""
        - แทนที่จะดูแค่ว่าโมเดลไหนทายว่า "ติด" หรือ "ไม่ติด" นำความน่าจะเป็นจากทั้ง 3 โมเดลมาหาค่าเฉลี่ยเพื่อให้ผลลัพธ์ที่เสถียรกว่า
        
        **4. ผลลัพธ์และการบันทึกโมเดล (Training Results)**
        - โมเดล Ensemble ชุดนี้สามารถทำความแม่นยำได้สูงถึง 92.85% ในการทดสอบ
        - ทำการบันทึกเป็นไฟล์ `f1_ensemble_model_v3.pkl` เพื่อนำมาใช้รันบนหน้าเว็บ Streamlit
        """)
    
    col_space_left, col_image, col_space_right = st.columns([1, 2, 1])

    with col_image:
        try:
            img_ens = BASE_DIR / 'result_ensemble.png'
            # ใส่ caption และสั่งให้ใช้ความกว้างเต็มพื้นที่คอลัมน์กลาง
            st.image(str(img_ens), caption='Confusion Matrix ของ Ensemble Model', use_container_width=True)
        except FileNotFoundError:
            st.error("❌ หาไฟล์รูปไม่เจอ!")
    
    st.success("### 4. แหล่งอ้างอิงข้อมูล (Data Sources)")
    st.markdown(dedent("""

    1. Scikit-learn Documentation: Ensemble Methods  
       https://scikit-learn.org/stable/modules/ensemble.html

    2. XGBoost Documentation  
       https://xgboost.readthedocs.io

    3. LightGBM Documentation  
       https://lightgbm.readthedocs.io

    4. Akash Rane. (2024). Formula 1 Pit Stop Dataset. Kaggle  
       https://www.kaggle.com/datasets/akashrane2609/formula-1-pit-stop-dataset

    5. Usman. (2024). Formula 1 Race Dataset for Predictive Analysis. Kaggle  
       https://www.kaggle.com/datasets/usman136/formula-1-race-dataset-for-predictive-analysis
"""))

# ====================== TAB 3: Model 2 Explanation ======================
with tabs[2]:
    st.markdown("### Model 2: Neural Network")
    st.write(""" แนวทางการพัฒนาโมเดล Neural Network (Deep Learning)
    """)
    
    st.success("### 1. การเตรียมข้อมูล (Data Preparation & Feature Engineering)")
    st.markdown("""
    * **Feature Selection**: คัดเลือกตัวแปรนำเข้า (Input Features) ทั้งหมด 9 ตัวแปรที่ผ่านการตรวจสอบแล้วว่าไม่มี Data Leakage
    * **Normalization (MinMaxScaler)**: เนื่องจาก Neural Network ใช้การคำนวณผ่าน Gradient หากข้อมูลมีสเกลที่ต่างกันมาก (เช่น Year เทียบกับ Start Position) จะทำให้โมเดลหาจุดที่เหมาะสมที่สุด (Convergence) ได้ยาก จึงปรับข้อมูลทุกตัวให้อยู่ในช่วง [0, 1]
    * **Handling Categorical Data**: แปลงชื่อสนาม ทีม และนักแข่ง ด้วยเทคนิค Label Encoding เพื่อเปลี่ยนข้อความเป็นดัชนีตัวเลขที่โครงข่ายประสาทสามารถนำไปคำนวณทางคณิตศาสตร์ได้
    """)
    
    st.success("### 2. ทฤษฎีอัลกอริทึม (Ensemble Theory)")
    st.markdown("""
    โมเดลที่พัฒนาขึ้นคือ Multi-Layer Perceptron (MLP) ซึ่งเป็นโครงข่ายประสาทเทียมแบบป้อนไปข้างหน้า (Feedforward Neural Network) โดยมีองค์ประกอบทางทฤษฎีดังนี้:
    """)
    
    st.markdown("""
    * **Architecture (โครงสร้าง)**
        * Input Layer: รับข้อมูล 9 มิติ
        * ปรับช่วงข้อมูลเป็น [0, 1] ด้วย **MinMaxScaler** เพื่อป้องกันปัญหาตัวแปรที่มีค่าสูง (เช่น ปี ค.ศ.) ข่มตัวแปรสำคัญอื่น ๆ
        * Output Layer: 1 นิวรอน พร้อมฟังก์ชัน Sigmoid เพื่อพ่นค่าความน่าจะเป็นในช่วง $0$ ถึง $1$
    * **Activation Functions**
        * ReLU (Rectified Linear Unit): ใช้ใน Hidden Layers เพื่อลดปัญหา Vanishing Gradient และช่วยให้โมเดลเรียนรู้ได้เร็วขึ้น
        * Sigmoid: ใช้ในชั้นสุดท้ายเพื่อจำแนกประเภท (Binary Classification)
    * **Regularization (Dropout)** ใส่ค่า Dropout 20% เพื่อสุ่มปิดนิวรอนบางส่วนระหว่างการเทรน ป้องกันปัญหา Overfitting (การที่โมเดลจำข้อสอบได้แม่นแต่ทำข้อสอบจริงไม่ได้)
    """)
    
    st.write("**ReLU (Rectified Linear Unit)**")
    st.latex(r"f(x) = \max(0, x)")
    
    st.write("**Sigmoid**")
    st.latex(r"\sigma(x) = \frac{1}{1 + e^{-x}}")
    
    st.success("### 3. ขั้นตอนการพัฒนาโมเดล (Model Development)")
    col_nn1, col_nn2 = st.columns(2)
    
    with col_nn1:
       st.markdown("""
        **1. การออกแบบโครงสร้าง (Network Architecture)**
        - ใช้โครงสร้างแบบ Sequential ซึ่งประกอบด้วยเลเยอร์ทั้งหมด 4 ชั้น
             * `Input Layer`: รับข้อมูลนำเข้าทั้ง 9 Feature
             * `Hidden Layers`: ประกอบด้วย 3 ชั้นหลัก ขนาด 64, 32 และ 16 นิวรอนตามลำดับ เพื่อเพิ่มความสามารถในการเรียนรู้ความสัมพันธ์ที่ซับซ้อน
             * `Output Layer`: มี 1 นิวรอน พร้อมฟังก์ชัน Sigmoid เพื่อทำนายความน่าจะเป็นของการติด Top 10
        
        **2. ฟังก์ชันกระตุ้นและระบบป้องกัน (Activation & Regularization)**
        * `ReLU (Rectified Linear Unit)`: ใช้ใน Hidden Layers ทุกชั้นเพื่อให้โมเดลเรียนรู้แบบ Non-linear ได้ดี และช่วยลดปัญหา Vanishing Gradient
        * `Sigmoid`: ใช้ในชั้นสุดท้ายเพื่อแปลงค่าเป็นความน่าจะเป็นสำหรับการจำแนกประเภท
        * `Dropout (0.2)`: แทรกหลังเลเยอร์แรก โดยการสุ่มปิดนิวรอน 20% ระหว่างเทรน เพื่อป้องกันปัญหา Overfitting หรือการที่โมเดลจดจำข้อมูลชุดเทรนมากเกินไปจนทำข้อสอบจริงไม่ได้
    
        """)
        
    with col_nn2:
        
        st.markdown("**4. การตั้งค่าการเรียนรู้ (Compilation)**")
        st.write("**Loss Function: Binary Crossentropy**")
        st.latex(r"L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]")
        st.markdown("""
        
        * `Loss Function`: ใช้ binary_crossentropy ซึ่งเหมาะสำหรับงานคัดแยก 2 ประเภท
        * `Optimizer`: ใช้ Adam Optimizer ซึ่งเป็นอัลกอริทึมที่ปรับค่าน้ำหนัก (Weights) ได้อย่างมีประสิทธิภาพและรวดเร็ว
        """)
        
        st.markdown("""
        **3. ผลลัพธ์และการบันทึกโมเดล (Training & Performance)**
        * `Process`: กำหนดการเทรนทั้งหมด 30 รอบ (Epochs) โดยใช้ Batch Size ขนาด 32
        * `Performance`: โมเดลนี้ทำความแม่นยำได้ประมาณ 90.19% ในชุดข้อมูลทดสอบ
        * `Model Persistence`: บันทึกโมเดลในรูปแบบไฟล์ f1_nn_model_v3.h5 เพื่อนำไปโหลดใช้งานบน Streamlit ต่อไป
        """)
    
    col_space_left, col_image, col_space_right = st.columns([1, 2, 1])

    with col_image:
        try:
            img_ens = BASE_DIR / 'result_neural_network.png'
            # ใส่ caption และสั่งให้ใช้ความกว้างเต็มพื้นที่คอลัมน์กลาง
            st.image(str(img_ens), caption='Confusion Matrix ของ Neural Network Model', use_container_width=True)
        except FileNotFoundError:
            st.error("❌ หาไฟล์รูปไม่เจอ!")
    
    st.success("### 4. แหล่งอ้างอิงข้อมูล (Data Sources)")
    st.markdown("""

    1. TensorFlow & Keras Documentation: Sequential Model  
    https://www.tensorflow.org/guide/keras

    2. Akash Rane. (2024). Formula 1 Pit Stop Dataset. Kaggle.  
    https://www.kaggle.com/datasets/akashrane2609/formula-1-pit-stop-dataset

    3. Usman. (2024). Formula 1 Race Dataset for Predictive Analysis. Kaggle.  
    https://www.kaggle.com/datasets/usman136/formula-1-race-dataset-for-predictive-analysis

    4. Chollet, F. (2017). Deep Learning with Python. Manning Publications.(สำหรับแนวคิดและทฤษฎีเบื้องหลังโครงข่ายประสาทเทียม)
    """)

# ====================== TAB 4: Test Model 1 (Ensemble) ======================
with tabs[3]:
    st.markdown("### ทดสอบ Model 1 — Ensemble ML")
    st.write("กรอกข้อมูลการแข่ง แล้วกดปุ่มเพื่อให้ Ensemble Model ทำนายโอกาสติด Top 10")

    col1, col2 = st.columns(2)
    with col1:
        year = st.slider('Season Year', int(min(m['seasons'])), int(max(m['seasons'])), int(max(m['seasons'])), key='ens_year')
        # season-aware GP list with fallback to global list
        gp_options = m['circuits_by_season'].get(year, m['global_gps']) or m['global_gps']
        sel_gp = st.selectbox('Grand Prix (Circuit)', gp_options, key='ens_gp')
        sel_wea = st.selectbox('Weather', m['weather'], key='ens_wea')
    with col2:
        # season-aware drivers and teams
        driver_options = m['drivers_by_season'].get(year, m['global_drivers']) or m['global_drivers']
        team_options = m['teams_by_season'].get(year, m['global_teams']) or m['global_teams']

        sel_driver = st.selectbox('Driver', driver_options, key='ens_driver')

        # if we have driver->teams mapping for the season, prefer that
        teams_for_driver = m['teams_by_season_and_driver'].get((year, sel_driver), [])
        if teams_for_driver:
            sel_team = st.selectbox('Team', teams_for_driver, key='ens_team')
        else:
            sel_team = st.selectbox('Team', team_options, key='ens_team')

        start_pos = st.number_input('Starting Position', 1, 20, 10, key='ens_start')
        qual_pos = st.number_input('Qualifying Position', 1, 20, 10, key='ens_qual')
        sel_tire = st.selectbox('Tire Strategy', m['tires'], key='ens_tire')

    # Data integrity checks and warnings
    if sel_driver not in driver_options:
        st.warning(f"Selected driver '{sel_driver}' not present in season {year}. Showing nearest available: {', '.join(driver_options[:5])}...")
    if sel_team not in team_options and (year, sel_driver) not in m['teams_by_season_and_driver']:
        st.warning(f"Selected team '{sel_team}' not present in season {year}.")

    if st.button('วิเคราะห์ด้วย Ensemble Model', type='primary'):
        # Prepare input using global lists
        gp_encoded = encode_selection(sel_gp, m['global_gps'])
        input_features = np.array([[ 
            (year - 1950) / 74.0, gp_encoded, gp_encoded,
            encode_selection(sel_wea, m['weather']),
            encode_selection(sel_driver, m['global_drivers']),
            encode_selection(sel_team, m['global_teams']),
            (start_pos - 1) / 19.0, (qual_pos - 1) / 19.0,
            encode_selection(sel_tire, m['tires'])
        ]])

        # RF probability (for component view)
        try:
            rf_prob = float(rf_model.predict_proba(input_features)[0][1])
        except Exception:
            try:
                rf_prob = float(np.squeeze(rf_model.predict(input_features)))
            except Exception:
                rf_prob = 0.0

        # Voting ensemble probability: prefer voting_model if available
        if voting_model is not None:
            try:
                ens_prob = float(voting_model.predict_proba(input_features)[0][1])
            except Exception:
                # fallback: average available estimators' probs or use rf_prob
                try:
                    preds = []
                    for name, est in getattr(voting_model, 'named_estimators', {}).items():
                        try:
                            p = float(est.predict_proba(input_features)[0][1])
                        except Exception:
                            p = float(np.squeeze(est.predict(input_features)))
                        preds.append(p)
                    ens_prob = float(np.mean(preds)) if len(preds) > 0 else rf_prob
                except Exception:
                    ens_prob = rf_prob
        else:
            # no voting model saved: use RF as proxy
            ens_prob = rf_prob

        # Render result section (show only Ensemble as main result)
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader('Result — Voting Ensemble')
        col_main = st.columns(1)[0]
        col_main.metric('Voting Ensemble (RF+GB+LR)', f"{ens_prob*100:.1f}%")

        def _mini_bar(pct, color):
            pct_clamped = max(0.0, min(1.0, pct))
            return f"""
            <div style='background:#0b1220;border-radius:8px;padding:6px'>
              <div style='background:#111827;border-radius:6px;padding:4px'>
                <div style='width:{pct_clamped*100:.1f}%;background:{color};height:12px;border-radius:6px'></div>
              </div>
            </div>
            """

        col_main.markdown(_mini_bar(ens_prob, '#10b981'), unsafe_allow_html=True)

        if ens_prob > 0.5:
            st.success(f"AI มั่นใจว่า {sel_driver} จะติด Top 10 แน่นอน! (Voting Ensemble)")
        else:
            st.warning(f"{sel_driver} ไม่แน่นอนว่าจะติด Top 10 อาจจะต้องเหนื่อยหน่อยใน Race นี้ (Voting Ensemble)")

        st.markdown('</div>', unsafe_allow_html=True)

# ====================== TAB 5: Test Model 2 (Neural Network) ======================
with tabs[4]:
    st.markdown("### ทดสอบ Model 2 — Neural Network")
    st.write("กรอกข้อมูลการแข่ง แล้วกดปุ่มเพื่อให้ Neural Network ทำนาย")

    col1, col2 = st.columns(2)
    with col1:
        year = st.slider('Season Year', int(min(m['seasons'])), int(max(m['seasons'])), int(max(m['seasons'])), key='nn_year')
        gp_options = m['circuits_by_season'].get(year, m['global_gps']) or m['global_gps']
        sel_gp = st.selectbox('Grand Prix (Circuit)', gp_options, key='nn_gp')
        sel_wea = st.selectbox('Weather', m['weather'], key='nn_wea')
    with col2:
        driver_options = m['drivers_by_season'].get(year, m['global_drivers']) or m['global_drivers']
        team_options = m['teams_by_season'].get(year, m['global_teams']) or m['global_teams']

        sel_driver = st.selectbox('Driver', driver_options, key='nn_driver')

        teams_for_driver = m['teams_by_season_and_driver'].get((year, sel_driver), [])
        if teams_for_driver:
            sel_team = st.selectbox('Team', teams_for_driver, key='nn_team')
        else:
            sel_team = st.selectbox('Team', team_options, key='nn_team')

        start_pos = st.number_input('Starting Position', 1, 20, 10, key='nn_start')
        qual_pos = st.number_input('Qualifying Position', 1, 20, 10, key='nn_qual')
        sel_tire = st.selectbox('Tire Strategy', m['tires'], key='nn_tire')

    if sel_driver not in driver_options:
        st.warning(f"Selected driver '{sel_driver}' not present in season {year}.")
    if sel_team not in team_options and (year, sel_driver) not in m['teams_by_season_and_driver']:
        st.warning(f"Selected team '{sel_team}' not present in season {year}.")

    if st.button('วิเคราะห์ด้วย Neural Network', type='primary'):
        gp_encoded = encode_selection(sel_gp, m['global_gps'])
        input_features = np.array([[ 
            (year - 1950) / 74.0, gp_encoded, gp_encoded,
            encode_selection(sel_wea, m['weather']),
            encode_selection(sel_driver, m['global_drivers']),
            encode_selection(sel_team, m['global_teams']),
            (start_pos - 1) / 19.0, (qual_pos - 1) / 19.0,
            encode_selection(sel_tire, m['tires'])
        ]])

        try:
            nn_pred = nn_model.predict(input_features)
            nn_prob = float(np.squeeze(nn_pred))
        except Exception:
            nn_prob = 0.0

        # Render result section (show only Neural Network)
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader('Result — Neural Network')
        col_main = st.columns(1)[0]
        col_main.metric('Neural Network', f"{nn_prob*100:.1f}%")

        def _mini_bar(pct, color):
            pct_clamped = max(0.0, min(1.0, pct))
            return f"""
            <div style='background:#0b1220;border-radius:8px;padding:6px'>
              <div style='background:#111827;border-radius:6px;padding:4px'>
                <div style='width:{pct_clamped*100:.1f}%;background:{color};height:12px;border-radius:6px'></div>
              </div>
            </div>
            """

        col_main.markdown(_mini_bar(nn_prob, '#3b82f6'), unsafe_allow_html=True)

        if nn_prob > 0.5:
            st.success(f"AI (NN) มั่นใจว่า {sel_driver} จะติด Top 10 แน่นอน!")
        else:
            st.warning(f"AI (NN): {sel_driver} ไม่แน่นอนว่าจะติด Top 10 อาจจะต้องเหนื่อยหน่อยใน Race นี้")

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Tip: เปลี่ยนตัวเลือกในแถบด้านข้างแล้วกดปุ่มเพื่อวิเคราะห์</div>', unsafe_allow_html=True)