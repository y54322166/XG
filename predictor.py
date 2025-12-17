# 导入 StreamLit 库，用于构建 Web 应用
import streamlit as st

# 导入 joblib 库，用于加载和保存机器学习模型
import joblib

# 导入 NumPy 库，用于数值计算
import numpy as np

# 导入 Pandas 库，用于数据处理和操作
import pandas as pd

# 导入 SHAP 库，用于解释机器学习模型的预测
import shap

# 导入 Matplotlib 库，用于数据可视化
import matplotlib.pyplot as plt


# 加载训练好的随机森林模型（XGBoost.pkl）
model = joblib.load('XGBoost.pkl')

# 从 X-test.csv 文件加载测试数据，以便用于 LIME 解释器
X_test = pd.read_csv('X-test.csv')  # 修正：变量名改为下划线

# 定义特征名称，对应数据集中的列名
feature_names = [
    "SSA",     # 比表面积
    "PV",     # 总孔体积
    "Vme",     # 介孔体积
    "Vmi",     # 微孔体积
    "RT",     # 温度
    "P",     # 压力
    "C",     # 碳含量
    "N",     # 氮含量
    "O",    # 氧含量
    "Pre",   # 前驱体物质
    "Mod",   # 改性方法
]

# Streamlit 用户界面
st.title("CO₂ Adsorption Capacity Predictor")  # 设置网页标题

# 比表面积：数值输入框
SSA = st.number_input("SSA,m²/g)", min_value=0.0, max_value=5000.0, value=1000.0, step=10.0)

# 总孔体积：数值输入框
PV = st.number_input("(PV, cm³/g)", min_value=0.0, max_value=1.58, value=0.5, step=0.0001)

# 介孔体积：数值输入框
Vme = st.number_input("(Vme, cm³/g)", min_value=0.0, max_value=0.67, value=0.3, step=0.0001)

# 微孔体积：数值输入框
Vmi = st.number_input("(Vmi, cm³/g)", min_value=0.0, max_value=1.07, value=0.2, step=0.0001)

# 温度 (RT, ℃)：数值输入框
RT = st.number_input("(RT,℃)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)

# 压强 (P, bar)：数值输入框
P = st.number_input("(P, bar) ", 
                    min_value=0.0, 
                    max_value=50.0, 
                    value=1.0, 
                    step=0.1,  # 保持步进0.1但调整标签说明
                    help="建议使用0.1-5.0 bar范围进行实验")

# 碳含量 (C, %)：数值输入框
C = st.number_input("(C, %)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)  # 修正：变量名改为大写C以保持一致性

# 氮含量 (N, %)：数值输入框
N = st.number_input("(N, %)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)  # 修正：变量名改为大写N以保持一致性

# 氧含量 (O, %)：数值输入框
O = st.number_input("(O, %)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)  # 修正：变量名改为大写O以保持一致性

# 前驱体类型 (Pre)：分类选择框（0-130）
Pre = st.selectbox("(Pre)：", options=range(0, 131))  # 包含0到130

# 改性方法（Mod）：分类选择框（0-9）
Mod = st.selectbox("(Mod)：", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 处理输入数据并进行预测
feature_values = [SSA, PV, Vme, Vmi, RT, P, C, N, O, Pre, Mod]  # 将用户输入的特征值存入列表
features = np.array([feature_values])  # 将特征转换为 NumPy 数组，适用于模型输入

# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("Predict", use_container_width=True):
    
    if model is None:
        st.error("Prediction unavailable: model failed to load successfully")
    else:
        with st.spinner("Calculating prediction results..."):
            # 预测吸附量
            predicted_value = model.predict(features)[0]
            
            # 尝试获取预测概率（如果模型支持）
            try:
                predicted_proba = model.predict_proba(features)[0]
                has_proba = True
            except:
                has_proba = False
            
            # 显示预测结果
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"Predicted Adsorption Capacity:")
                st.markdown(f"# {predicted_value:.2f} mmol/g")
            
            with col2:
                if has_proba:
                    probability = predicted_proba[1] * 100 if len(predicted_proba) > 1 else predicted_proba[0] * 100
                    st.write(f"Model Confidence:")
                    st.markdown(f"# {probability:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 根据预测结果生成建议
            st.markdown("Material Performance Evaluation")
            
            if predicted_value > 5.0:
                st.success("Excellent adsorbent material: predicted high CO₂ adsorption capacity with strong CO₂ capture potential")
                st.info("Recommendation: This material is well-suited for industrial CO₂ capture applications")
            elif predicted_value > 3.0:
                st.warning("Good adsorbent material: predicted moderate CO₂ adsorption capacity with appreciable CO₂ uptake capability")
                st.info("Recommendation: Further optimization of the material structure may be considered to enhance adsorption performance")
            else:
                st.error("Fair adsorbent material: predicted low CO₂ adsorption capacity with limited CO₂ uptake capability")
                st.info("Recommendation: Adjusting the material composition or synthesis route is advised to improve adsorption performance")
            
            # SHAP 解释
            st.markdown(" SHAP 解释")
            
            try:
                # 创建SHAP解释器
                explainer_shap = shap.TreeExplainer(model)
                
                # 计算SHAP值
                shap_values = explainer_shap.shap_values(features)
                
                # 创建图表
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 创建SHAP force plot
                shap.force_plot(
                    explainer_shap.expected_value,
                    shap_values[0] if len(shap_values.shape) == 2 else shap_values,
                    pd.DataFrame([feature_values], columns=feature_names),
                    matplotlib=True,
                    show=False
                )
                
                plt.title("SHAP Force Plot - Feature Contribution Visualization", fontsize=14, pad=20)
                plt.tight_layout()
                
                # 保存并显示图像
                plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
                st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')
                
            except Exception as e:
                # 捕获所有异常并显示错误信息
                st.error(f"生成SHAP解释图时出错: {str(e)}")
                st.info("请检查: 1) 模型是否支持SHAP解释 2) 特征数据格式是否正确")
                
            finally:
                # 无论是否发生异常都执行清理
                plt.close('all')  # 关闭所有matplotlib图形，释放内存
