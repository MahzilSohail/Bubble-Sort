import streamlit as st
import random
import numpy as np
from sklearn.cluster import KMeans

st.set_page_config(page_title="Bubble Sort", page_icon="🫧", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}
.game-title {
    font-family: 'Fredoka One', cursive;
    font-size: 2.6rem;
    text-align: center;
    background: linear-gradient(90deg, #f9c74f, #f94144, #43aa8b, #577590);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    letter-spacing: 2px;
}
.subtitle {
    text-align: center;
    color: #aaa;
    font-size: 0.88rem;
    margin-bottom: 1rem;
}
.jar-card {
    background: rgba(255,255,255,0.06);
    border: 2px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 16px 12px 12px;
    text-align: center;
    transition: border-color 0.2s, box-shadow 0.2s;
    min-height: 300px;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.jar-card.selected {
    border-color: #f9c74f !important;
    box-shadow: 0 0 18px rgba(249,199,79,0.45) !important;
}
.jar-card.target {
    border-color: #43aa8b !important;
    box-shadow: 0 0 18px rgba(67,170,139,0.35) !important;
}
.jar-card.full-jar {
    border-color: #f94144 !important;
}
.jar-title {
    font-family: 'Fredoka One', cursive;
    font-size: 1.1rem;
    color: #ddd;
    margin-bottom: 10px;
    letter-spacing: 1px;
}
.bubble-column {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    flex: 1;
    justify-content: flex-end;
    padding-bottom: 8px;
}
.bubble {
    width: 62px;
    height: 62px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    box-shadow: inset -6px -6px 12px rgba(0,0,0,0.3),
                inset 4px 4px 8px rgba(255,255,255,0.35),
                0 4px 14px rgba(0,0,0,0.4);
}
.bubble-red   { background: radial-gradient(circle at 35% 35%, #ff6b6b, #c0392b); }
.bubble-blue  { background: radial-gradient(circle at 35% 35%, #74b9ff, #2980b9); }
.bubble-green { background: radial-gradient(circle at 35% 35%, #55efc4, #00b894); }
.top-bubble   { border: 3px solid rgba(255,255,255,0.65); }
.empty-slot {
    width: 62px; height: 62px;
    border-radius: 50%;
    border: 2px dashed rgba(255,255,255,0.1);
}
.capacity-info {
    color: rgba(255,255,255,0.35);
    font-size: 0.72rem;
    margin-top: 8px;
}
.stat-row {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin: 0.5rem 0 1rem;
    flex-wrap: wrap;
}
.stat-chip {
    background: rgba(255,255,255,0.08);
    border-radius: 30px;
    padding: 5px 18px;
    color: #ddd;
    font-size: 0.85rem;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.15);
}
.stat-chip span { color: #f9c74f; }
.step-box {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 14px;
    padding: 10px 20px;
    text-align: center;
    color: #eee;
    font-size: 0.95rem;
    margin: 0 auto 1rem;
}
.step-box strong { color: #f9c74f; }
.win-banner {
    background: linear-gradient(135deg, #43aa8b, #2d9648);
    border-radius: 18px;
    padding: 20px 30px;
    text-align: center;
    color: white;
    font-family: 'Fredoka One', cursive;
    font-size: 1.8rem;
    letter-spacing: 1px;
    box-shadow: 0 8px 30px rgba(67,170,139,0.5);
    margin: 0.5rem auto 1rem;
}
.held-info {
    background: rgba(249,199,79,0.12);
    border: 2px dashed #f9c74f;
    border-radius: 50px;
    padding: 6px 20px;
    color: #f9c74f;
    font-size: 0.9rem;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 0.6rem;
}
/* Override streamlit button styles */
.stButton > button {
    background: rgba(255,255,255,0.07) !important;
    color: #eee !important;
    border: 1.5px solid rgba(255,255,255,0.2) !important;
    border-radius: 30px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    padding: 6px 10px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    margin-top: 8px !important;
}
.stButton > button:hover {
    background: rgba(249,199,79,0.18) !important;
    border-color: #f9c74f !important;
    color: #f9c74f !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
COLORS    = ["red", "blue", "green"]
EMOJI     = {"red": "🔴", "blue": "🔵", "green": "🟢"}
JAR_NAMES = ["Jar A", "Jar B", "Jar C", "Buffer"]
MAX_CAP   = 3

# ── K-Means helper ─────────────────────────────────────────────────────────────
def kmeans_hint(jars):
    all_balls = [b for jar in jars for b in jar]
    if not all_balls:
        return "No balls to analyse."
    color_vec = {"red": [1,0,0], "blue": [0,1,0], "green": [0,0,1]}
    X  = np.array([color_vec[b] for b in all_balls])
    n  = min(3, len(set(all_balls)))
    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    km.fit(X)
    cluster_map = {}
    for c in range(n):
        idxs = [i for i, l in enumerate(km.labels_) if l == c]
        dom  = max(COLORS, key=lambda col: sum(1 for i in idxs if all_balls[i] == col))
        cluster_map[c] = dom
    counts = {cluster_map[c]: int(np.sum(km.labels_ == c)) for c in range(n)}
    parts  = [f"{EMOJI[col]} {cnt} **{col}**" for col, cnt in counts.items()]
    return "🤖 K-Means found: " + " · ".join(parts) + " — move same-colour bubbles into one jar each!"

# ── Game init ──────────────────────────────────────────────────────────────────
def init_game():
    balls = COLORS * 3
    random.shuffle(balls)
    st.session_state.jars         = [balls[0:3], balls[3:6], balls[6:9], []]
    st.session_state.selected_jar = None
    st.session_state.held_ball    = None
    st.session_state.moves        = 0
    st.session_state.won          = False
    st.session_state.hint_text    = ""

if "jars" not in st.session_state:
    init_game()

def check_win():
    jars = st.session_state.jars
    
    completed_colors = []

    for jar in jars:
        # Only consider full jars
        if len(jar) == 3:
            if len(set(jar)) == 1: 
                completed_colors.append(jar[0])
            else:
                return False 

    # Must have exactly 3 completed jars with 3 unique colors
    return len(completed_colors) == 3 and len(set(completed_colors)) == 3
# ─────────────────────────────────────────────────────────────────────────────
#  RENDER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="game-title">Bubble Sort</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sort bubbles by colour · K-Means powered hints</div>',
            unsafe_allow_html=True)

moves = st.session_state.moves
st.markdown(f"""
<div class="stat-row">
  <div class="stat-chip">Moves: <span>{moves}</span></div>
  <div class="stat-chip">🔴 Red &nbsp; 🔵 Blue &nbsp; 🟢 Green</div>
  <div class="stat-chip">Max per jar: <span>3</span></div>
</div>
""", unsafe_allow_html=True)

# Win banner
if st.session_state.won:
    st.markdown(
        f'<div class="win-banner">🎉 Brilliant! Sorted in {moves} moves! 🎉</div>',
        unsafe_allow_html=True)

# Instruction strip
sel = st.session_state.selected_jar
if not st.session_state.won:
    if sel is None:
        st.markdown(
            '<div class="step-box">👆 <strong>Step 1 —</strong> '
            'Click a <em>"Pick"</em> button below a jar to grab its top bubble</div>',
            unsafe_allow_html=True)
    else:
        col = st.session_state.held_ball
        st.markdown(
            f'<div style="text-align:center">'
            f'<div class="held-info">{EMOJI[col]} Holding a {col} bubble</div>'
            f'</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="step-box"><strong>Step 2 —</strong> '
            'Click <em>"Drop here"</em> under the destination jar &nbsp;|&nbsp; '
            'or <em>"Cancel"</em> to put it back</div>',
            unsafe_allow_html=True)

# ── 4 Jar columns ──────────────────────────────────────────────────────────────
jars  = st.session_state.jars
cols4 = st.columns(4)

for idx in range(4):
    jar = jars[idx]
    with cols4[idx]:

        # Card CSS class
        card_cls = "jar-card"
        if sel is not None:
            card_cls += " selected" if sel == idx else " target"
        elif len(jar) >= MAX_CAP:
            card_cls += " full-jar"

        # Bubbles HTML — render slots top-down so bottom of jar = last item
        # We show MAX_CAP slots; occupied ones show bubble, empty ones show ghost
        bubbles_html = '<div class="bubble-column">'
        # Pad to MAX_CAP for visual stability
        padded = [""] * (MAX_CAP - len(jar)) + list(jar)  # empties on top
        for slot_i, ball in enumerate(padded):
            actual_pos = slot_i - (MAX_CAP - len(jar))  # position in actual jar
            is_top     = (ball != "" and actual_pos == 0)
            if ball:
                top_cls = " top-bubble" if is_top else ""
                bubbles_html += (
                    f'<div class="bubble bubble-{ball}{top_cls}">{EMOJI[ball]}</div>'
                )
            else:
                bubbles_html += '<div class="empty-slot"></div>'
        bubbles_html += '</div>'

        st.markdown(
            f'<div class="{card_cls}">'
            f'<div class="jar-title">{JAR_NAMES[idx]}</div>'
            f'{bubbles_html}'
            f'<div class="capacity-info">{len(jar)} / {MAX_CAP}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Buttons under each jar ─────────────────────────────────────────────
        if not st.session_state.won:
            sel_now = st.session_state.selected_jar  # re-read each iter

            if sel_now is None:
                # Phase 1 — pick source
                if st.button(
                    f"Pick {JAR_NAMES[idx]}" if jar else "— empty —",
                    key=f"btn_pick_{idx}",
                    disabled=(len(jar) == 0)
                ):
                    st.session_state.selected_jar = idx
                    st.session_state.held_ball    = jar[0]
                    st.rerun()

            elif sel_now == idx:
                # Cancel
                if st.button("❌ Cancel", key=f"btn_cancel_{idx}"):
                    st.session_state.selected_jar = None
                    st.session_state.held_ball    = None
                    st.rerun()

            else:
                # Phase 2 — drop destination
                is_full = len(jar) >= MAX_CAP
                if st.button(
                    "Full 🚫" if is_full else f"Drop here",
                    key=f"btn_drop_{idx}",
                    disabled=is_full
                ):
                    ball = jars[sel_now].pop(0)
                    jars[idx].insert(0, ball)
                    st.session_state.moves        += 1
                    st.session_state.selected_jar  = None
                    st.session_state.held_ball     = None
                    if check_win():
                        st.session_state.won = True
                    st.rerun()

# ── Hint text ──────────────────────────────────────────────────────────────────
if st.session_state.hint_text:
    st.info(st.session_state.hint_text)

# ── Action buttons ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)
with b1:
    if st.button("🔀 New Game", key="action_new"):
        init_game()
        st.rerun()
with b2:
    if st.button("💡 K-Means Hint", key="action_hint"):
        st.session_state.hint_text = kmeans_hint(jars)
        st.rerun()
with b3:
    if st.button("🔄 Reshuffle", key="action_shuffle"):
        init_game()
        st.rerun()

# ── How to play ────────────────────────────────────────────────────────────────
with st.expander("📖 How to Play"):
    st.markdown("""
<style>
    .white-text {
        color: white;
    }
    .white-text table {
        color: white;
    }
    .white-text th, .white-text td {
        color: white;
    }
</style>

<div class="white-text">

**Goal:** Sort all 9 bubbles so each non-empty jar contains only **one colour**.

| Step | Action |
|------|--------|
| 1️⃣ | Click **"Pick Jar X"** below any jar to grab its **top bubble** |
| 2️⃣ | Click **"Drop here"** under the destination jar |
| ❌ | Click **"Cancel"** under the selected jar to put the bubble back |

**Rules:**
- Maximum **3 bubbles** per jar
- Use the **Buffer** jar as temporary storage
- The top bubble (brightest border) is always moved first

**K-Means Hint:** Click 💡 to run unsupervised ML clustering on all balls — it tells you exactly how many of each colour remain so you can plan your moves!

</div>
""", unsafe_allow_html=True)