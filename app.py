import streamlit as st
import pandas as pd
import numpy as np
import random
import math
import pulp

# --- 1. ページ基本設定 ---
st.set_page_config(
    page_title="文化祭シフト最適化システム",
    page_icon="🏫",
    layout="wide"
)

# ==========================================
# 2. 人工データ生成および最適化関数の定義
# ==========================================
def run_shift_optimization(num_students=100, num_times=15, num_departments=17, seed=42):
    """
    人工データを生成し、PuLPを用いてシフト最適化を実行する関数
    """
    rng = random.Random(seed)

    # 1. 集合の定義
    students = [f"S{s + 1:03d}" for s in range(num_students)]
    times = [f"T{t + 1:02d}" for t in range(num_times)]
    departments = [f"部門{r + 1:02d}" for r in range(num_departments)]
    grades = [1, 2, 3]
    roles = ["一般", "役職者"]

    # 2. 時間帯ラベル（30分刻み）
    time_labels = {}
    start_minutes = 9 * 60
    for time_index, t in enumerate(times):
        begin = start_minutes + 30 * time_index
        end = begin + 30
        time_labels[t] = f"{begin // 60:02d}:{begin % 60:02d}-{end // 60:02d}:{end % 60:02d}"

    # 3. 生徒属性
    student_grade = {s: rng.choices(grades, weights=[0.34, 0.33, 0.33], k=1)[0] for s in students}
    student_role = {s: ("役職者" if rng.random() < 0.18 else "一般") for s in students}

    # 4. 必要人数 N_{t,r}
    department_base_required = {r: rng.randint(2, 4) for r in departments}
    required = {}
    for t in times:
        for r in departments:
            required[t, r] = max(1, department_base_required[r] + rng.choice([-1, 0, 1]))

    # 5. 配置可能性 a_{s,t}
    available = {(s, t): 0 for s in students for t in times}
    for s in students:
        available_count = rng.randint(6, 13)
        for t in rng.sample(times, min(available_count, len(times))):
            available[s, t] = 1

    # 6. 希望得点
    preference_score = {(s, t, r): 0 for s in students for t in times for r in departments}
    for s in students:
        for t in times:
            if available[s, t] == 1:
                selected_depts = rng.sample(departments, min(2, len(departments)))
                preference_score[s, t, selected_depts[0]] = 2
                if len(selected_depts) > 1:
                    preference_score[s, t, selected_depts[1]] = 1

    # ==========================================
    # PuLP モデルの構築と求解
    # ==========================================
    model = pulp.LpProblem("HighSchoolShift", pulp.LpMaximize)

    active_keys = [(s, t, r) for s in students for t in times for r in departments if available[s, t] == 1]
    x = {
        (s, t, r): pulp.LpVariable(f"x_{s}_{t}_{r}", cat=pulp.LpBinary)
        for s, t, r in active_keys
    }

    def get_x(s, t, r):
        return x.get((s, t, r), 0)

    # 目的関数：希望得点の最大化
    preference_objective = pulp.lpSum(
        preference_score[s, t, r] * var for (s, t, r), var in x.items()
    )
    model.setObjective(preference_objective)

    # 制約1：配置人数
    staff_shortage = {
        (t, r): pulp.LpVariable(f"shortage_{t}_{r}", lowBound=0, cat=pulp.LpInteger)
        for t in times for r in departments
    }
    for t in times:
        for r in departments:
            model += (
                pulp.lpSum(get_x(s, t, r) for s in students) + staff_shortage[t, r] == required[t, r],
                f"staff_{t}_{r}"
            )

    # 制約2：同一時間帯の重複禁止
    for s in students:
        for t in times:
            model += (
                pulp.lpSum(get_x(s, t, r) for r in departments) <= 1,
                f"no_overlap_{s}_{t}"
            )

    # 求解の実行
    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)
    status = pulp.LpStatus[model.status]

    # 結果の抽出
    assignments = []
    if status in ["Optimal", "Feasible"]:
        for (s, t, r), var in x.items():
            if pulp.value(var) > 0.5:
                score = preference_score[s, t, r]
                assignments.append({
                    "生徒": s,
                    "学年": student_grade[s],
                    "役職": student_role[s],
                    "時間帯": time_labels[t],
                    "部門": r,
                    "希望順位": {2: "第1希望", 1: "第2希望", 0: "希望外"}[score]
                })

    df_assignments = pd.DataFrame(assignments)
    return status, df_assignments, required, time_labels, departments, students


# ==========================================
# 3. サイドバーの設定 (Sidebar Controls)
# ==========================================
st.sidebar.header("⚙️ シミュレーション設定")
num_students = st.sidebar.slider("生徒数 (Students)", min_value=50, max_value=150, value=100, step=10)
seed_val = st.sidebar.number_input("乱数シード (Seed)", value=7, step=1)

# --- 側边栏：实时展示当前Seed和参数下生成的随机制约条件 ---
st.sidebar.markdown("---")
st.sidebar.subheader("📋 乱数で生成された制約条件")

# 临时用同等的随机逻辑生成预览数据，使侧边栏能够动态展示
temp_rng = random.Random(int(seed_val))
num_departments = 17
num_times = 15
departments = [f"部門{r + 1:02d}" for r in range(num_departments)]
times = [f"T{t + 1:02d}" for t in range(num_times)]

# 生成各部门基础需求预览矩阵
dept_base_req = {r: temp_rng.randint(2, 4) for r in departments}
preview_data = {}
for t in times[:5]:  # 为了侧边栏美观，展示前5个时间段
    preview_data[t] = [max(1, dept_base_req[r] + temp_rng.choice([-1, 0, 1])) for r in departments]

df_preview = pd.DataFrame(preview_data, index=departments)
df_preview.index.name = "部門名"
df_preview.reset_index(inplace=True)

st.sidebar.text(f"対象生徒数: {num_students} 人")
st.sidebar.text(f"総部門数: {num_departments} 箇所")
st.sidebar.text(f"総時間帯数: {num_times} コマ")

with st.sidebar.expander("🔍 各部門の必要人数（プレビュー）"):
    st.markdown("現在のシード値で生成された各部門・時間帯別の必要人数（抜粋）：")
    st.dataframe(df_preview, use_container_width=True)

st.sidebar.text(f"生徒の稼働可能時間: ランダム適用済み (Seed: {seed_val})")


# ==========================================
# 4. メイン画面の操作と実行
# ==========================================
st.title("🏫 文化祭シフト最適化システム (Streamlit版)")
st.markdown("高校生のシフト希望と各部門の条件を考慮して、数理最適化（PuLP）により最適なシフトを自動生成します。")

# 最適化計算の実行ボタン
if st.button("🚀 最適化計算を実行する"):
    with st.spinner("数理モデルを求解中です。少々お待ちください..."):
        status, df_result, required_dict, time_labels, depts, students_list = run_shift_optimization(
            num_students=num_students, seed=int(seed_val)
        )

    st.success(f"求解完了！ ステータス: {status}")

    if not df_result.empty:
        st.subheader("📋 生成されたシフト結果（プレビュー）")
        st.dataframe(df_result.head(20), use_container_width=True)

        # ダウンロードボタンの提供
        csv = df_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 シフト結果をCSVでダウンロード",
            data=csv,
            file_name='shift_optimization_result.csv',
            mime='text/csv',
        )
    else:
        st.warning("有効な解が見つかりませんでした。パラメータを調整してください。")
else:
    st.info("左側のサイドバーで生徒数やシード値を調整し、「最適化計算を実行する」ボタンを押してください。")
