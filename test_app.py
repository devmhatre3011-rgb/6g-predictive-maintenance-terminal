import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Professional page settings
st.set_page_config(page_title="Thales 6G Analytics Terminal", layout="wide")

st.title("🛰️ Industry 4.0: 6G Predictive Maintenance & Analytics Terminal")
st.markdown("---")

data_file = "Thales_Group_Manufacturing.csv"

if not os.path.exists(data_file):
    st.error(f"❌ Critical Error: Data tracking core file '{data_file}' not found.")
else:
    # High-performance pipeline execution cache
    @st.cache_data
    def execute_data_pipeline(path):
        df = pd.read_csv(path)
        
        # Enforce strict chronological order across timelines
        df = df.sort_values(by=["Machine_ID", "Date", "Timestamp"]).reset_index(drop=True)
        
        # 1. Feature Engineering
        df["Rolling_Temp_Avg"] = df.groupby("Machine_ID")["Temperature_C"].transform(lambda x: x.rolling(10, min_periods=1).mean())
        df["Sensor_Dev_Temp"] = df["Temperature_C"] - df["Rolling_Temp_Avg"]
        df["Vibration_Power_Ratio"] = df["Vibration_Hz"] / (df["Power_Consumption_kW"] + 0.1)
        
        # 2. Machine Learning Anomaly Detection (Isolation Forest)
        model_features = ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', 
                          'Network_Latency_ms', 'Sensor_Dev_Temp', 'Vibration_Power_Ratio']
        
        iso_forest = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
        iso_forest.fit(df[model_features])
        anomaly_distances = iso_forest.decision_function(df[model_features])
        
        # Standardize score from 0 (Healthy) to 100 (Critical Outlier)
        df["Anomaly_Score"] = (1 - (anomaly_distances - anomaly_distances.min()) / (anomaly_distances.max() - anomaly_distances.min())) * 100
        
        # 3. Risk Tier Mapping
        def assign_risk_tier(score):
            if score < 45: return "🟢 Low Risk (Normal)"
            elif score < 70: return "🟡 Medium Risk (Monitor)"
            else: return "🔴 High Risk (Action Required)"
                
        df["Risk_Tier"] = df["Anomaly_Score"].apply(assign_risk_tier)
        return df

    with st.spinner("Processing data transformations & running Isolation Forest ML Model..."):
        df = execute_data_pipeline(data_file)
    
    # ==========================================
    # 🎛️ SIDEBAR INTERACTIVE CONTROL DEEP-DIVE
    # ==========================================
    st.sidebar.header("🔍 Specialized Asset Deep-Dive")
    st.sidebar.markdown("Isolate and audit a single machine's health timeline history track.")
    
    unique_machines = sorted(df["Machine_ID"].unique())
    selected_machine = st.sidebar.selectbox("Select Target Machine ID:", unique_machines)
    
    machine_timeline = df[df["Machine_ID"] == selected_machine].copy()
    
    st.sidebar.markdown(f"### Current Status: **{selected_machine}**")
    latest_log = machine_timeline.iloc[-1]
    st.sidebar.metric("Latest Anomaly Score", f"{latest_log['Anomaly_Score']:.1f} / 100")
    st.sidebar.markdown(f"Risk Tier Evaluation: \n**{latest_log['Risk_Tier']}**")

    # Clean corporate layout navigation
    menu_tabs = st.tabs([
        "📋 Fleet Executive Summary", 
        "📊 Telemetry Distribution Profiles", 
        "📡 6G Network Health Audits",
        "⚙️ Predictive Feature Vectors",
        "🚨 ML Operational Risk Desk",
        "📈 Chronological Asset Timeline"
    ])
    
    # === SECTION 1: FLEET EXECUTIVE SUMMARY ===
    with menu_tabs[0]:
        st.subheader("⚙️ Global Asset Operational Matrices")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Parsed Telemetry Packets", f"{len(df):,}")
        m_col2.metric("Monitored Industrial Machine Assets", f"{df['Machine_ID'].nunique()}")
        m_col3.metric("Data Integrity Anomalies", f"{df.isnull().sum().sum()}")
        
        layout_col1, layout_col2 = st.columns(2)
        with layout_col1:
            st.markdown("#### Distribution of Fleet Operational Modes")
            st.dataframe(df['Operation_Mode'].value_counts(), use_container_width=True)
        with layout_col2:
            st.markdown("#### Realized Factory Production Efficiency Breakdown")
            st.dataframe(df['Efficiency_Status'].value_counts(), use_container_width=True)

    # === SECTION 2: TELEMETRY DISTRIBUTION PROFILES ===
    with menu_tabs[1]:
        st.subheader("📐 Physical Hardware Sensor Variances & Operational Baselines")
        sensor_columns = ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW']
        st.dataframe(df[sensor_columns].describe().round(2), use_container_width=True)
        
        fig_sensor, axes_sensor = plt.subplots(1, 3, figsize=(16, 4.5))
        sns.set_theme(style="whitegrid")
        for i, col_name in enumerate(sensor_columns):
            sns.histplot(data=df, x=col_name, kde=True, color="teal", bins=40, ax=axes_sensor[i])
            axes_sensor[i].axvline(df[col_name].mean(), color='crimson', linestyle='--', linewidth=2, label='Mean Baseline')
            axes_sensor[i].set_title(f"{col_name} Frequency", fontweight="bold")
            axes_sensor[i].legend(loc="upper right")
        plt.tight_layout()
        st.pyplot(fig_sensor)
        plt.close()

    # === SECTION 3: 6G NETWORK HEALTH AUDITS ===
    with menu_tabs[2]:
        st.subheader("🌐 Evaluation of High-Speed 6G Low-Latency Signal Paths")
        fig_network, axes_network = plt.subplots(1, 2, figsize=(16, 5))
        sns.scatterplot(data=df.head(2000), x='Network_Latency_ms', y='Error_Rate_%', hue='Efficiency_Status', ax=axes_network[0], alpha=0.6)
        axes_network[0].set_title("6G Signal Path Propagation Delay vs. Process Error Escalations", fontweight="bold")
        sns.boxplot(data=df, x='Efficiency_Status', y='Packet_Loss_%', palette='viridis', ax=axes_network[1])
        axes_network[1].set_title("Packet Loss Profiles Mapped Against Efficiency Outputs", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_network)
        plt.close()

    # === SECTION 4: PREDICTIVE FEATURE VECTORS (UPDATED WITH SECTION 3 GRAPHS!) ===
    with menu_tabs[3]:
        st.subheader("🛠️ Custom Engineered Indicators for Early Anomaly Detection")
        log_columns = ["Machine_ID", "Temperature_C", "Rolling_Temp_Avg", "Sensor_Dev_Temp", "Vibration_Power_Ratio"]
        st.dataframe(df[log_columns].head(15), use_container_width=True)
        
        st.markdown("#### 📈 Probability Densities of Pre-Failure Markers")
        
        # New code blocks to generate the density distributions for Section 3
        fig_features, axes_features = plt.subplots(1, 2, figsize=(16, 4.5))
        sns.set_theme(style="whitegrid")
        
        # Panel 1: Sensor_Dev_Temp Density distribution
        sns.histplot(data=df, x="Sensor_Dev_Temp", kde=True, color="purple", bins=40, ax=axes_features[0])
        axes_features[0].set_title("Thermal Deviation Profile (Sensor_Dev_Temp)", fontweight="bold")
        
        # Panel 2: Vibration_Power_Ratio Skewed distribution
        sns.histplot(data=df, x="Vibration_Power_Ratio", kde=True, color="chocolate", bins=40, ax=axes_features[1])
        axes_features[1].set_title("Kinetic Instability Footprint (Vibration_Power_Ratio)", fontweight="bold")
        
        plt.tight_layout()
        st.pyplot(fig_features)
        plt.close()

    # === SECTION 5: ML OPERATIONAL RISK DESK ===
    with menu_tabs[4]:
        st.subheader("🚨 Real-Time Unsupervised Anomaly Risk Assessment")
        kpi1, kpi2, kpi3 = st.columns(3)
        risk_counts = df["Risk_Tier"].value_counts()
        kpi1.metric("Healthy Fleet Units (Low)", f"{risk_counts.get('🟢 Low Risk (Normal)', 0):,}")
        kpi2.metric("Assets Under Strain (Medium)", f"{risk_counts.get('🟡 Medium Risk (Monitor)', 0):,}")
        kpi3.metric("Critical Pre-Failure Warnings (High)", f"{risk_counts.get('🔴 High Risk (Action Required)', 0):,}")
        
        st.markdown("---")
        selected_tier = st.selectbox("Filter Logs by Operational Risk Level:", 
                                     ["All Records", "🔴 High Risk (Action Required)", "🟡 Medium Risk (Monitor)", "🟢 Low Risk (Normal)"])
        
        display_cols = ["Machine_ID", "Date", "Timestamp", "Temperature_C", "Vibration_Hz", "Network_Latency_ms", "Anomaly_Score", "Risk_Tier"]
        if selected_tier == "All Records":
            filtered_df = df[display_cols].head(100)
        else:
            filtered_df = df[df["Risk_Tier"] == selected_tier][display_cols].head(100)
            
        def color_risk_rows(row):
            if "🔴 High Risk" in str(row["Risk_Tier"]): return ["background-color: rgba(220, 53, 69, 0.25)"] * len(row)
            elif "🟡 Medium Risk" in str(row["Risk_Tier"]): return ["background-color: rgba(255, 193, 7, 0.2)"] * len(row)
            return [""] * len(row)
            
        st.dataframe(filtered_df.style.apply(color_risk_rows, axis=1).format("{:.2f}", subset=["Anomaly_Score"]), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📈 Multidimensional Outlier Separation View")
        
        # Plot showing how the Isolation Forest successfully isolated high risk elements
        fig_ml, ax_ml = plt.subplots(figsize=(14, 5))
        sns.scatterplot(
            data=df.head(3000), 
            x='Temperature_C', 
            y='Vibration_Power_Ratio', 
            hue='Risk_Tier', 
            palette={'🟢 Low Risk (Normal)': 'seagreen', '🟡 Medium Risk (Monitor)': 'orange', '🔴 High Risk (Action Required)': 'crimson'},
            alpha=0.7,
            ax=ax_ml
        )
        ax_ml.set_title("Machine Learning Risk Classification Boundaries (Temperature vs. Instability Footprint)", fontweight="bold")
        st.pyplot(fig_ml)
        plt.close()

    # === SECTION 6: CHRONOLOGICAL ASSET TIMELINE ===
    with menu_tabs[5]:
        st.subheader(f"📈 Predictive Health Trajectory Analysis for Asset: {selected_machine}")
        
        # AUTOMATED PRESCRIPTIVE MAINTENANCE DIRECTIVE ENGINE
        st.markdown("### 🦾 Automated Diagnostic Dispatch Brief")
        
        if "🔴 High Risk" in latest_log["Risk_Tier"]:
            st.error(f"⚠️ **CRITICAL ACTION MANDATED FOR {selected_machine}**\n\n"
                     f"**Diagnostic Trigger:** The ML engine detected a critical anomaly score of **{latest_log['Anomaly_Score']:.1f}/100**.\n\n"
                     f"**Root-Cause Vectors:** Thermal delta tracking indicates an uncharacteristic deviation (`Sensor_Dev_Temp`: {latest_log['Sensor_Dev_Temp']:.2f}°C) "
                     f"coupled with an elevated kinetic imbalance signature (`Vibration_Power_Ratio`: {latest_log['Vibration_Power_Ratio']:.2f}).\n\n"
                     f"**Prescriptive Directive:** Immediately dispatch floor technician to inspect mechanical bearings for alignment shifts and check structural coupling degradation.")
        elif "🟡 Medium Risk" in latest_log["Risk_Tier"]:
            st.warning(f"⚡ **ELEVATED MONITORING STATUS FOR {selected_machine}**\n\n"
                       f"**Diagnostic Trigger:** Machine learning variance thresholds have crossed the safety boundary with a score of **{latest_log['Anomaly_Score']:.1f}/100**.\n\n"
                       f"**Prescriptive Directive:** Schedule a secondary physical calibration check within the next 48 hours. Ensure local 6G data transmission transceiver links are not dropping telemetry frames.")
        else:
            st.success(f"✅ **ASSET STATUS OPTIMAL FOR {selected_machine}**\n\n"
                       f"**Diagnostic Trigger:** Current Anomaly Score is safely stabilized at **{latest_log['Anomaly_Score']:.1f}/100**.\n\n"
                       f"**Prescriptive Directive:** No tactical maintenance required. Asset is executing within calibrated 6G-factory control parameters.")
        
        st.markdown("---")
        
        # Build line graph tracking anomaly history
        fig_timeline, ax_timeline = plt.subplots(figsize=(16, 4.5))
        sns.set_theme(style="whitegrid")
        
        plot_timeline = machine_timeline.tail(150) 
        plot_timeline["Timeline_Label"] = plot_timeline["Date"] + " " + plot_timeline["Timestamp"]
        
        sns.lineplot(data=plot_timeline, x="Timeline_Label", y="Anomaly_Score", ax=ax_timeline, color="navy", linewidth=2.5, marker="o", markersize=4)
        ax_timeline.axhline(70, color="crimson", linestyle="--", linewidth=2, label="🔴 High Risk Threshold (70+)")
        ax_timeline.axhline(45, color="orange", linestyle="-.", linewidth=1.5, label="🟡 Medium Risk Threshold (45-70)")
        
        ax_timeline.set_title(f"Continuous Real-Time Anomaly Score Trajectory [{selected_machine}]", fontsize=14, fontweight="bold", pad=15)
        ax_timeline.set_xlabel("Chronological Log Timestamp Sequences (Last 150 Records)", fontsize=11, labelpad=10)
        ax_timeline.set_ylabel("Computed Machine Learning Anomaly Score", fontsize=11)
        ax_timeline.set_ylim(0, 100)
        ax_timeline.legend(loc="upper left")
        
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12)) 
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig_timeline)
        plt.close()