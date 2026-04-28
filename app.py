import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
from datetime import datetime

# 1. إعدادات الصفحة الاحترافية
st.set_page_config(page_title="منظومة توقع الطاقة", layout="wide")

# 2. تعريب أعمدة الداتا سيت الجديدة
translation_dict = {
    "Device_Type": "نوع الجهاز",
    "Usage_Hours_Per_Day": "ساعات التشغيل اليومية",
    "Power_Rating_Watts": "قدرة الجهاز (وات)",
    "Efficiency_Factor": "معامل كفاءة الطاقة",
    "Days_Per_Month": "أيام العمل في الشهر",
    "Region": "النطاق الجغرافي",
    "Sector": "نوع القطاع"
}

# 3. ستايل الواجهة (CSS)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px; border-radius: 20px; color: white; text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1); margin-bottom: 25px;
    }
    .stNumberInput, .stSelectbox { border-radius: 10px !important; }
    .result-card {
        background: white; padding: 25px; border-radius: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05); border-right: 10px solid #1e3c72;
    }
    .alert-card {
        background: #fff5f5; padding: 25px; border-radius: 20px;
        border: 2px solid #ff4b4b; text-align: center; color: #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# 4. تحميل البيانات
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('realistic_egypt_electricity_dataset.csv')
        return [col for col in df.columns if col != 'Monthly_Cost_EGP']
    except:
        return list(translation_dict.keys())

features = load_data()

# --- واجهة المستخدم ---
st.markdown('<div class="main-header"><h1>⚡ منظومة التوقع الذكي لتكلفة الكهرباء</h1><p>إدارة الطاقة بناءً على البيانات الواقعية</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ الإعدادات المتقدمة")
    kwh_price = st.slider("سعر الكيلو وات (ج.م)", 0.5, 3.0, 1.45)
    st.markdown("---")
    st.error("🛑 حد التنبيه: 1,000 ج.م")

st.subheader("📥 إدخال معايير الاستهلاك:")
user_inputs = {}
cols = st.columns(3)
for i, feat in enumerate(features):
    with cols[i % 3]:
        label = translation_dict.get(feat, feat)
        user_inputs[feat] = st.number_input(label, value=0.0, format="%.2f")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("📊 تشغيل نموذج التوقع"):
    # معادلة حسابية احترافية بناءً على أعمدة الداتا الجديدة
    # (Watts * Hours * Days) / 1000 * Price
    total_kwh = (user_inputs.get('Power_Rating_Watts', 0) * user_inputs.get('Usage_Hours_Per_Day', 0) * user_inputs.get('Days_Per_Month', 0)) / 1000
    
    final_egp = total_kwh * kwh_price
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if final_egp > 1000:
            st.markdown(f'<div class="alert-card"><h2>⚠️ تجاوز الميزانية: {final_egp:,.2f} ج.م</h2><p>الاستهلاك مرتفع جداً</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-card"><h2 style="color:#1e3c72;">✅ التكلفة التقديرية: {final_egp:,.2f} ج.م</h2><p>ضمن النطاق المسموح</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.write("📈 **إحصائيات إضافية**")
        st.metric("إجمالي الكيلو وات", f"{total_kwh:,.1f}")

    # الرسم البياني
    st.markdown("---")
    months = ['يناير', 'فبراير', 'مارس', 'أبريل', 'مايو', 'يونيو', 'يوليو', 'أغسطس', 'سبتمبر', 'أكتوبر', 'نوفمبر', 'ديسمبر']
    # محاكاة التغير الموسمي
    trend = [final_egp * (1 + 0.2 * np.sin(i/1.5)) for i in range(12)]
    fig = px.area(x=months, y=trend, title="توقع الاستهلاك على مدار السنة", labels={'x':'الشهر', 'y':'التكلفة (ج.م)'})
    fig.update_traces(line_color='#1e3c72', fillcolor='rgba(30, 60, 114, 0.1)')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr><center>منظومة ذكاء الأعمال للطاقة 2026</center>", unsafe_allow_html=True)
