# -*- coding: utf-8 -*-
"""
v19 全艇スコア解析アプリ（中穴〜大穴特化・波乱予想AI 搭載版）
"""
import re
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import lightgbm as lgb
from bs4 import BeautifulSoup

JST = timezone(timedelta(hours=+9), 'JST')
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
req_session = requests.Session()
req_session.headers.update(UA)
adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=3)
req_session.mount('https://', adapter)
req_session.mount('http://', adapter)

JCD_NAME = {
    1:"桐生", 2:"戸田", 3:"江戸川", 4:"平和島", 5:"多摩川", 6:"浜名湖",
    7:"蒲郡", 8:"常滑", 9:"津", 10:"三国", 11:"びわこ", 12:"住之江",
    13:"尼崎", 14:"鳴門", 15:"丸亀", 16:"児島", 17:"宮島", 18:"徳山",
    19:"下関", 20:"若松", 21:"芦屋", 22:"福岡", 23:"唐津", 24:"大村"
}

@dataclass
class Racer:
    name: str
    age: int; cls_val: int; weight: int; f_count: int; avg_st: float
    n_win: float; n_2ren: float; l_win: float
    l_2ren: float; m_2ren: float; b_2ren: float

def calc_extended_stats(racers: List[Racer]) -> List[Dict]:
    avg_win = sum(r.n_win for r in racers) / 6.0
    avg_motor = sum(r.m_2ren for r in racers) / 6.0
    avg_st = sum(r.avg_st for r in racers) / 6.0
    win_rates = sorted([r.n_win for r in racers], reverse=True)
    motors = sorted([r.m_2ren for r in racers], reverse=True)
    sts = sorted([r.avg_st for r in racers])
    stats = []
    for i, r in enumerate(racers):
        st_diff_in = round(r.avg_st - racers[i-1].avg_st, 3) if i > 0 else 0.0
        win_diff_in = round(r.n_win - racers[i-1].n_win, 2) if i > 0 else 0.0
        st_diff_out = round(r.avg_st - racers[i+1].avg_st, 3) if i < 5 else 0.0
        win_diff_out = round(r.n_win - racers[i+1].n_win, 2) if i < 5 else 0.0
        stats.append({
            "win_dev": round(r.n_win - avg_win, 2), "motor_dev": round(r.m_2ren - avg_motor, 4), "st_dev": round(avg_st - r.avg_st, 3),
            "win_rank": win_rates.index(r.n_win) + 1, "motor_rank": motors.index(r.m_2ren) + 1, "st_rank": sts.index(r.avg_st) + 1,
            "st_diff_in": st_diff_in, "win_diff_in": win_diff_in, "st_diff_out": st_diff_out, "win_diff_out": win_diff_out
        })
    return stats

@st.cache_resource
def load_lgb_model(filename: str):
    try: return lgb.Booster(model_file=filename)
    except: return None

# 穴党特化モデルを読み込み
lgb_model   = load_lgb_model('lgb_score_v19_ana.txt')
prob1_model = load_lgb_model('lgb_p1_v19_ana.txt')
prob2_model = load_lgb_model('lgb_p2_v19_ana.txt')
prob3_model = load_lgb_model('lgb_p3_v19_ana.txt')

def get_lgb_features(r: Racer, lane: int, venue: str, rel: Dict) -> list:
    jcd = {v: k for k, v in JCD_NAME.items()}.get(venue, 1)
    return [
        float(jcd), float(lane), float(r.cls_val), float(r.age), float(r.weight), float(r.f_count), float(r.avg_st),
        float(r.n_win), float(r.n_2ren), float(r.l_win), float(r.l_2ren), float(r.m_2ren), float(r.b_2ren),
        float(rel["win_dev"]), float(rel["motor_dev"]), float(rel["st_dev"]), float(rel["win_rank"]), float(rel["motor_rank"]), float(rel["st_rank"]),
        float(rel["st_diff_in"]), float(rel["win_diff_in"]), float(rel["st_diff_out"]), float(rel["win_diff_out"])
    ]

def rank_all(racers: List[Racer], venue: str) -> Tuple[List[Dict], Optional[float]]:
    out = []
    rel_stats = calc_extended_stats(racers)
    for i, r in enumerate(racers):
        lane = i + 1
        features = get_lgb_features(r, lane, venue, rel_stats[i])
        ai_score, p1, p2, p3 = 0.0, 0.0, 0.0, 0.0
        if lgb_model:   ai_score = round(lgb_model.predict([features])[0] * 10, 2)
        if prob1_model: p1 = round(prob1_model.predict([features])[0] * 100, 1)
        if prob2_model: p2 = round(prob2_model.predict([features])[0] * 100, 1)
        if prob3_model: p3 = round(prob3_model.predict([features])[0] * 100, 1)
      
        out.append({"lane": lane, "racer": r, "score": ai_score, "1着率": p1, "2着率": p2, "3着率": p3, "rel": rel_stats[i]})
    out.sort(key=lambda x: x["score"], reverse=True)
    lane1_prob = next((x["1着率"] for x in out if x["lane"] == 1), None)
    return out, lane1_prob

def get_bet_probs(ranked: List[Dict]) -> List[Dict]:
    # 波乱確率そのままを使用するため正規化は行わない
    bet_probs = []
    for r1 in ranked:
        for r2 in ranked:
            if r1["lane"] == r2["lane"]: continue
        
            for r3 in ranked:
                if r3["lane"] in (r1["lane"], r2["lane"]): continue
                
                # 穴展開になる確率 ＝ (1着波乱確率 * 2着波乱確率 * 3着波乱確率) / 10000
                prob = (r1["1着率"] * r2["2着率"] * r3["3着率"]) / 10000
       
                bet_probs.append({"bet": f'{r1["lane"]}-{r2["lane"]}-{r3["lane"]}', "prob": round(prob, 2)})
                
    bet_probs.sort(key=lambda x: x["prob"], reverse=True)
    return bet_probs

def fetch_kyotei24_data(jcd: int, rno: int, dstr: str):
    url = f"https://info.kyotei.fun/info-{dstr}-{jcd:02d}-{rno}.html"
    try:
        r = req_session.get(url, timeout=7)
        r.encoding = r.apparent_encoding
        html = r.text if r.status_code == 200 else ""
    except: return None
    if not html or "出走表" not in html: return None
    try: soup = BeautifulSoup(html, "lxml")
    except: soup = BeautifulSoup(html, "html.parser")
    lane_to_rank = {}
    jyuni_divs = soup.find_all('div', class_='jyuni')
    has_result = False
    if len(jyuni_divs) >= 6:
        for i in range(6):
            txt = jyuni_divs[i].get_text(strip=True)
            if txt.isdigit(): lane_to_rank[i+1] = txt; has_result = True
    payoff = 0
    if has_result:
        payoff_div = soup.find('div', class_='race_result_end_label', string=re.compile('3連単'))
        if payoff_div and payoff_div.parent:
            money_span = payoff_div.parent.find('span', class_='race_result_end_money_num')
            if money_span:
                ptxt = money_span.get_text(strip=True).replace(',', '')
                if ptxt.isdigit(): payoff = int(ptxt)
    rd = [{"name": f"選手{i+1}", "age": 30, "cls": 1, "weight": 50, "f": 0, "st": 0.17, "nw": 0.0, "n2": 0.0, "lw": 0.0, "l2": 0.0, "m2": 0.0, "b2": 0.0} for i in range(6)]
    cls_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    current_label = ""
    for tr in soup.find_all('tr'):
        tds = tr.find_all(['td', 'th'])
        if not tds: continue
        if len(tds) >= 7: current_label = tds[0].get_text(strip=True).replace('\n', '').replace(' ', ''); data_tds = tds[-6:]
        elif len(tds) == 6 and current_label: data_tds = tds
        else: current_label = ""; continue
        for i in range(6):
            txt_raw = data_tds[i].get_text(" ", strip=True)
            txt_nospace = txt_raw.replace(' ', '').replace('　', '').replace('\n', '')
            if "選手名" in current_label:
                m_age = re.search(r'\((\d{2})\)', txt_nospace)
                if m_age: rd[i]["age"] = int(m_age.group(1))
 
            elif "選手情報" in current_label or "支部" in current_label or "級" in current_label:
                m_cls = re.search(r'([A12B]{2})', txt_nospace)
                if m_cls: rd[i]["cls"] = cls_map.get(m_cls.group(1), 1)
                m_w = re.search(r'(\d+)kg', txt_nospace, re.IGNORECASE)
                if m_w: rd[i]["weight"] = int(m_w.group(1))
            elif "全国" in current_label and "勝率" in current_label:
                m_2 = re.search(r'^([\d\.]+)', txt_nospace); m_w = re.search(r'\(([\d\.]+)\)', txt_nospace)
                if m_2: rd[i]["n2"] = float(m_2.group(1))/100.0 if float(m_2.group(1))>1.0 else float(m_2.group(1))
                if m_w: rd[i]["nw"] = float(m_w.group(1))
            elif "当地" in current_label and "勝率" in current_label:
                m_2 = re.search(r'^([\d\.]+)', txt_nospace); m_w = re.search(r'\(([\d\.]+)\)', txt_nospace)
                if m_2: rd[i]["l2"] = float(m_2.group(1))/100.0 if float(m_2.group(1))>1.0 else float(m_2.group(1))
                if m_w: rd[i]["lw"] = float(m_w.group(1))
            elif "モータ" in current_label and "2連率" in current_label:
                m = re.search(r'^([\d\.]+)', txt_nospace)
               
                if m: rd[i]["m2"] = float(m.group(1))/100.0 if float(m.group(1))>1.0 else float(m.group(1))
            elif "ボート" in current_label and "2連率" in current_label:
                m = re.search(r'^([\d\.]+)', txt_nospace)
                if m: rd[i]["b2"] = float(m.group(1))/100.0 if float(m.group(1))>1.0 else float(m.group(1))
            elif "平均ST" in current_label:
              
                try: rd[i]["st"] = float(txt_nospace)
                except: pass
            elif "フライング" in current_label:
                try: rd[i]["f"] = int(txt_nospace)
                except: pass
    if sum(x["nw"] for x in rd) == 0: return None
    racers = [Racer(name=x["name"], age=x["age"], cls_val=x["cls"], weight=x["weight"], f_count=x["f"], avg_st=x["st"], n_win=x["nw"], n_2ren=x["n2"], l_win=x["lw"], l_2ren=x["l2"], m_2ren=x["m2"], b_2ren=x["b2"]) for x in rd]
    return racers, lane_to_rank, payoff, has_result

st.set_page_config(page_title="v19 穴党・波乱特化AI", layout="wide")
st.title("🚤 v19 全艇スコア解析 (穴党AI)")
st.caption("🚨 【波乱特化モデル】配当3000円未満のレースを全切りし、旨味のあるレースだけを狙撃します")

if not lgb_model:
    st.warning("⚠️ v19用の穴党AIモデルが見つかりません。")

tab1, tab2 = st.tabs(["🔍 1レース解析", "📊 穴党バックテスト"])

with tab1:
    col1, col2 = st.columns(2)
    with col1: d_input = st.date_input("日付", value=datetime.now(JST).date())
    with col2: v_idx = st.selectbox("場", options=list(JCD_NAME.keys()), format_func=lambda x: JCD_NAME[x])
    r_idx = st.selectbox("レース", options=list(range(1, 13)))
    if st.button("🔍 解析開始", type="primary", use_container_width=True):
        dstr = d_input.strftime("%Y%m%d")
        res = fetch_kyotei24_data(v_idx, r_idx, dstr)
        if res:
            racers, _, _, _ = res
            ranked, _ = rank_all(racers, JCD_NAME[v_idx])
            st.success("解析完了！")
            
            # このレースが荒れる確率（全買い目の確率の合計）
            bet_probs = get_bet_probs(ranked)
      
            total_rough_prob = sum(bp["prob"] for bp in bet_probs)
            
            if total_rough_prob >= 10.0:
                st.markdown(f"🔥 **【波乱注意報】** 穴展開になる確率: **<span style='color:red;'>{total_rough_prob:.1f}%</span>**", unsafe_allow_html=True)
            else:
                st.markdown(f"⚠️ **【ガチガチ推奨】** 荒れる気配なし（穴確率: {total_rough_prob:.1f}%）")

         
            df_disp = []
            for item in ranked:
                racer = item["racer"]; rel = item["rel"]
                df_disp.append({
                    "枠": item["lane"], "選手名": racer.name, "穴スコア": item["score"], 
                    "穴1着率(%)": item["1着率"], "穴2着率(%)": item["2着率"], "穴3着率(%)": item["3着率"],
                    "ST": racer.avg_st, "内ST差": f"{rel['st_diff_in']:+.3f}", "外ST差": f"{rel['st_diff_out']:+.3f}"
      
                })
            st.dataframe(pd.DataFrame(df_disp).set_index("枠"), use_container_width=True)
            
            st.subheader("🎯 旨味のある買い目 (上位10点)")
            for i, bp in enumerate(bet_probs[:10]):
                st.write(f"**第{i+1}位** {bp['bet']} : **{bp['prob']}%**")
        else:
        
            st.error("出走表が取得できませんでした。")

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1: bt_start = st.date_input("開始日 ", value=datetime.now(JST).date() - timedelta(days=1))
    with col2: bt_end = st.date_input("終了日 ", value=datetime.now(JST).date() - timedelta(days=1))
    with col3: bt_venue_idx = st.selectbox("場を指定", options=[0] + list(JCD_NAME.keys()), format_func=lambda x: "全国（すべて）" if x==0 else JCD_NAME[x])
    
    st.markdown("#### 🎯 穴党フィルター")
    st.write("※ 穴狙いのため確率は低く出ます。0.5%〜2.0%の間でスライダーを調整してください。")
    prob_threshold = st.slider("【購入ライン】買い目の穴確率が何%以上のものを買うか", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    if st.button("📊 穴党バックテスト実行", type="primary", use_container_width=True):
        days = [(bt_start + timedelta(days=i)).strftime("%Y%m%d") for i in range((bt_end - bt_start).days + 1)]
        matches = []
        prog = st.progress(0.0)
        tasks = [(dstr, j, r) for dstr in days for j in ([bt_venue_idx] if bt_venue_idx != 0 else list(range(1, 25))) for r in range(1, 13)]
        st.write(f"全 {len(tasks)} レースを解析中...")
        
        def analyze_race(d, j, r):
            res = fetch_kyotei24_data(j, r, d)
   
            if not res: return None
            racers, lane_to_rank, payoff, has_result = res
            venue_name = JCD_NAME.get(j, "不明")
            ranked, _ = rank_all(racers, venue_name)
            
            bet_probs = get_bet_probs(ranked)
            
            # 的中判定用の buy_bets と、表示用の buy_bets_disp を作成
            buy_bets = [bp["bet"] for bp in bet_probs if bp["prob"] >= prob_threshold]
            buy_bets_disp = [f"{bp['bet']}({bp['prob']}%)" for bp in bet_probs if bp["prob"] >= prob_threshold]
            
            if not has_result:
                hit_str, payoff_disp, actual_result, actual_prob_disp, hit_amount = "⏳", "-", "結果待ち" if buy_bets else "堅いので見送り", "-", 0
            else:
                actual_result = ""
                actual_prob_disp = "-"
       
                r1 = next((k for k, v in lane_to_rank.items() if str(v) == '1'), None)
                r2 = next((k for k, v in lane_to_rank.items() if str(v) == '2'), None)
                r3 = next((k for k, v in lane_to_rank.items() if str(v) == '3'), None)
                if r1 and r2 and r3: 
                    actual_result = f"{r1}-{r2}-{r3}"
                    
                    # 結果の買い目の確率のみを取得し、別変数に格納
                    actual_prob = next((bp["prob"] for bp in bet_probs if bp["bet"] == actual_result), None)
                    if actual_prob is not None:
                        actual_prob_disp = f"{actual_prob}%"

                if not buy_bets:
                    hit_str, payoff_disp, hit_amount = "ー", f"({payoff})", 0
                else:
                    hit = actual_result in buy_bets
               
                    hit_str, payoff_disp, hit_amount = ("🎯" if hit else "❌"), payoff, (payoff if hit else 0)
            
            return {
                "日付": d, "場": venue_name, "R": r,
                "買い目": ", ".join(buy_bets_disp) if buy_bets_disp else "見", "点数": len(buy_bets), 
                "結果": actual_result, "結果確率": actual_prob_disp, "的中": hit_str, "払戻金": payoff_disp, "_hit_amount": hit_amount
            }
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            future_to_task = {executor.submit(analyze_race, d, j, r): (d, j, r) for d, j, r in tasks}
            done_count = 0
            
            for future in concurrent.futures.as_completed(future_to_task):
                done_count += 1
                prog.progress(done_count / len(tasks) if len(tasks) > 0 else 1.0)
                res = future.result()
                if res: matches.append(res)
                   
        matches.sort(key=lambda x: (x["日付"], x["場"], x["R"]))
        if matches:
            df_bt = pd.DataFrame(matches)
            bet_races = [m for m in matches if m["点数"] > 0 and m["的中"] in ["🎯", "❌"]]
            hits = [m for m in bet_races if m["的中"] == "🎯"]
            if bet_races:
  
                total_invest = sum(m["点数"] for m in bet_races) * 100
                total_return = sum(m["_hit_amount"] for m in bet_races)
                hit_rate = len(hits) / len(bet_races) * 100
                ret_rate = total_return / total_invest * 100 if total_invest > 0 else 0
      
                st.success(f"🔥 勝負レース: {len(bet_races)}件 (ガチガチ回避: {len([m for m in matches if m['点数']==0])}件)")
                
                st.info(f"💰 **総投資**: {total_invest:,}円 / **総回収**: {total_return:,}円 (回収率: {ret_rate:.1f}% / **勝率(的中率)**: {hit_rate:.1f}%)")
                
            # 「結果確率」列を新たに追加
            disp_cols = ["日付", "場", "R", "買い目", "結果", "結果確率", "的中", "払戻金"]
            st.dataframe(df_bt[disp_cols], use_container_width=True)
