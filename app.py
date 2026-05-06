import html
import json
from io import BytesIO
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Cancel Order Report Generator",
    page_icon="📉",
    layout="wide",
)


# =========================
# Helpers
# =========================
REQUIRED_COLUMNS = ["Order ID", "Cancelled Time", "Cancel Reason"]
PREFERRED_TIME_COLS = ["Cancelled Time", "Created Time", "Paid Time"]


def clean_column_name(col: str) -> str:
    return str(col).replace("\ufeff", "").strip()


def stringify_id(x) -> str:
    """Keep long TikTok Order IDs safe as strings, without .0 / tabs / spaces."""
    if pd.isna(x):
        return ""
    s = str(x).replace("\t", "").strip()
    if s.endswith(".0") and s[:-2].replace(".", "", 1).isdigit():
        s = s[:-2]
    return s


def parse_datetime_series(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
        .str.replace("\t", "", regex=False)
        .str.replace("\r", "", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "NaT": np.nan})
    )
    # TikTok exports are usually like 04/05/2026 11:54:28 PM.
    # Try the exact format first to avoid ambiguous parsing, then fallback for edge cases.
    parsed = pd.to_datetime(cleaned, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    missing = parsed.isna() & cleaned.notna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(cleaned.loc[missing], errors="coerce")
    return parsed


def parse_money_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("\t", "", regex=False)
        .str.strip(),
        errors="coerce",
    ).fillna(0)


def pct(n, d, digits=1):
    if d == 0 or pd.isna(d):
        return 0.0
    return round(float(n) / float(d) * 100, digits)


def safe_div(n, d):
    return float(n) / float(d) if d else 0.0


def fmt_float(x, digits=1):
    if pd.isna(x):
        x = 0
    return f"{x:.{digits}f}"


def fmt_money(x):
    if pd.isna(x):
        x = 0
    return f"${x:,.2f}"


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        # Try common encodings. TikTok exports usually work with utf-8-sig.
        data = uploaded_file.getvalue()
        last_err = None
        for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
            try:
                return pd.read_csv(BytesIO(data), dtype=str, encoding=enc)
            except Exception as e:
                last_err = e
        raise last_err
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, dtype=str)
    raise ValueError("只支持 CSV / XLSX / XLS 文件")


def first_non_null(series: pd.Series):
    x = series.dropna()
    x = x[x.astype(str).str.strip() != ""]
    return x.iloc[0] if len(x) else np.nan


def mode_or_first(series: pd.Series):
    x = series.dropna().astype(str).str.strip()
    x = x[x != ""]
    if len(x) == 0:
        return "Unknown"
    mode = x.mode()
    return mode.iloc[0] if len(mode) else x.iloc[0]


def join_unique(series: pd.Series, max_items=20):
    vals = []
    seen = set()
    for v in series.dropna().astype(str):
        v = v.replace("\t", "").strip()
        if v and v not in seen:
            seen.add(v)
            vals.append(v)
        if len(vals) >= max_items:
            vals.append("...")
            break
    return "; ".join(vals)


def build_order_level(raw: pd.DataFrame, order_id_col: str) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [clean_column_name(c) for c in df.columns]
    order_id_col = clean_column_name(order_id_col)

    df[order_id_col] = df[order_id_col].apply(stringify_id)
    df = df[df[order_id_col] != ""].copy()

    # Parse all time columns that exist.
    for c in PREFERRED_TIME_COLS + ["RTS Time", "Shipped Time", "Delivered Time"]:
        if c in df.columns:
            df[f"__parsed_{c}"] = parse_datetime_series(df[c])

    # Numeric fields useful for optional business context.
    for c in ["Quantity", "Sku Quantity of return", "Order Amount", "Order Refund Amount", "SKU Subtotal After Discount"]:
        if c in df.columns:
            df[f"__num_{c}"] = parse_money_series(df[c])

    agg = {}
    for c in df.columns:
        if c == order_id_col:
            continue
        if c.startswith("__parsed_"):
            agg[c] = first_non_null
        elif c in ["Cancel Reason", "Cancel By", "Order Status", "Order Substatus", "Cancelation/Return Type", "Payment Method", "Fulfillment Type", "Delivery Option", "State", "Country"]:
            agg[c] = mode_or_first
        elif c.startswith("__num_Order Amount"):
            agg[c] = first_non_null  # Order Amount repeats on each SKU row.
        elif c.startswith("__num_Order Refund Amount"):
            agg[c] = "sum"           # Refund amount is usually SKU-level; sum to order-level.
        elif c.startswith("__num_Quantity") or c.startswith("__num_Sku Quantity") or c.startswith("__num_SKU Subtotal"):
            agg[c] = "sum"
        elif c in ["Seller SKU", "SKU ID", "Product Name", "Variation", "Product Category"]:
            agg[c] = join_unique
        else:
            agg[c] = first_non_null

    order_df = df.groupby(order_id_col, as_index=False).agg(agg)
    order_df = order_df.rename(columns={order_id_col: "Order ID"})

    # Friendly derived columns.
    if "Seller SKU" in df.columns:
        sku_count = df.groupby(order_id_col)["Seller SKU"].nunique(dropna=True).reset_index(name="SKU Count")
        sku_count[order_id_col] = sku_count[order_id_col].apply(stringify_id)
        sku_count = sku_count.rename(columns={order_id_col: "Order ID"})
        order_df = order_df.merge(sku_count, on="Order ID", how="left")
    else:
        order_df["SKU Count"] = np.nan

    if "__num_Quantity" in order_df.columns:
        order_df = order_df.rename(columns={"__num_Quantity": "Item Quantity"})
    if "__num_Order Amount" in order_df.columns:
        order_df = order_df.rename(columns={"__num_Order Amount": "Order Amount Parsed"})
    if "__num_Order Refund Amount" in order_df.columns:
        order_df = order_df.rename(columns={"__num_Order Refund Amount": "Order Refund Amount Parsed"})
    if "__parsed_Cancelled Time" in order_df.columns:
        order_df = order_df.rename(columns={"__parsed_Cancelled Time": "Cancelled Datetime"})
    if "__parsed_Created Time" in order_df.columns:
        order_df = order_df.rename(columns={"__parsed_Created Time": "Created Datetime"})
    if "__parsed_Paid Time" in order_df.columns:
        order_df = order_df.rename(columns={"__parsed_Paid Time": "Paid Datetime"})

    return order_df


def classify_live_segment(hour: int, live1_start: int, live1_end: int, live2_start: int, live2_end: int) -> str:
    if pd.isna(hour):
        return "Unknown"
    h = int(hour)
    if live1_start <= h < live1_end:
        return "直播①"
    if live2_start <= h < live2_end:
        return "直播②"
    return "非直播"


def prepare_analysis(order_df: pd.DataFrame, time_col: str, reason_col: str,
                     start_date: date, end_date: date,
                     live1_start: int, live1_end: int, live2_start: int, live2_end: int) -> pd.DataFrame:
    df = order_df.copy()
    df["__dt"] = df[time_col]
    df = df[df["__dt"].notna()].copy()
    df["Date"] = df["__dt"].dt.date
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
    df["Hour"] = df["__dt"].dt.hour.astype(int)
    df["Weekday Num"] = df["__dt"].dt.weekday
    df["Day Type"] = np.where(df["Weekday Num"] >= 5, "周末 Weekend", "工作日 Weekday")
    df["Live Segment"] = df["Hour"].apply(lambda h: classify_live_segment(h, live1_start, live1_end, live2_start, live2_end))
    df["Is Live"] = df["Live Segment"].isin(["直播①", "直播②"])
    df["Cancel Reason Clean"] = df[reason_col].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"})
    return df


def calendar_day_counts(start_date: date, end_date: date):
    days = pd.date_range(start=start_date, end=end_date, freq="D")
    weekday_days = int((days.weekday < 5).sum())
    weekend_days = int((days.weekday >= 5).sum())
    return len(days), weekday_days, weekend_days


def hourly_counts(df: pd.DataFrame, mask=None):
    if mask is None:
        s = df.groupby("Hour").size()
    else:
        s = df[mask].groupby("Hour").size()
    return [int(s.get(h, 0)) for h in range(24)]


def peak_label(counts):
    if not counts or max(counts) == 0:
        return "-", 0
    maxv = max(counts)
    hrs = [i for i, v in enumerate(counts) if v == maxv]
    if len(hrs) == 1:
        return f"{hrs[0]}点", maxv
    if len(hrs) <= 3:
        return " & ".join([f"{h}点" for h in hrs]), maxv
    return f"{hrs[0]}点等", maxv


def reason_table(df: pd.DataFrame, n=7):
    total = len(df)
    counts = df["Cancel Reason Clean"].value_counts().head(n)
    rows = []
    max_count = int(counts.max()) if len(counts) else 0
    for reason, cnt in counts.items():
        rows.append({
            "reason": str(reason),
            "count": int(cnt),
            "pct": pct(cnt, total),
            "bar": pct(cnt, max_count, 0) if max_count else 0,
        })
    return rows


def make_reason_rows(rows, color="var(--live1)"):
    if not rows:
        return '<div class="empty-note">暂无数据</div>'
    out = []
    for r in rows:
        out.append(f'''
          <div class="reason-row">
            <div class="reason-name" title="{html.escape(r['reason'])}">{html.escape(r['reason'])}</div>
            <div class="reason-bar-wrap"><div class="reason-bar" style="width:{r['bar']}%;background:{color}"></div></div>
            <div class="reason-cnt">{r['count']}</div>
            <div class="reason-pct">{r['pct']:.1f}%</div>
          </div>
        ''')
    return "\n".join(out)


def make_insights(df: pd.DataFrame, start_date: date, end_date: date,
                  live1_start: int, live1_end: int, live2_start: int, live2_end: int):
    total = len(df)
    total_days, weekday_days, weekend_days = calendar_day_counts(start_date, end_date)
    weekday_count = int((df["Day Type"] == "工作日 Weekday").sum())
    weekend_count = int((df["Day Type"] == "周末 Weekend").sum())
    weekday_avg = safe_div(weekday_count, weekday_days)
    weekend_avg = safe_div(weekend_count, weekend_days)

    live1_count = int((df["Live Segment"] == "直播①").sum())
    live2_count = int((df["Live Segment"] == "直播②").sum())
    live_count = live1_count + live2_count
    nonlive_count = total - live_count
    live_hours = max(0, live1_end - live1_start) + max(0, live2_end - live2_start)
    nonlive_hours = 24 - live_hours
    live_intensity = safe_div(live_count, live_hours)
    nonlive_intensity = safe_div(nonlive_count, nonlive_hours)
    intensity_multiple = safe_div(live_intensity, nonlive_intensity)

    all_counts = hourly_counts(df)
    pk_label, pk_val = peak_label(all_counts)

    wd_live_pct = pct(int(((df["Day Type"] == "工作日 Weekday") & df["Is Live"]).sum()), weekday_count)
    we_live_pct = pct(int(((df["Day Type"] == "周末 Weekend") & df["Is Live"]).sum()), weekend_count)

    top_reason = "-"
    top_reason_count = 0
    top_reason_pct = 0
    if total:
        vc = df["Cancel Reason Clean"].value_counts()
        top_reason = str(vc.index[0])
        top_reason_count = int(vc.iloc[0])
        top_reason_pct = pct(top_reason_count, total)

    payment_count = int(df["Cancel Reason Clean"].str.contains("payment", case=False, na=False).sum())
    address_count = int(df["Cancel Reason Clean"].str.contains("address", case=False, na=False).sum())
    mistake_count = int(df["Cancel Reason Clean"].str.contains("mistake|mistaken", case=False, na=False).sum())

    if weekend_days and weekday_days:
        diff_pct = pct(weekend_avg - weekday_avg, weekday_avg) if weekday_avg else 0
        if weekend_avg > weekday_avg:
            insight_1 = f"周末日均 <strong>{weekend_avg:.1f}单/天</strong>，比工作日 {weekday_avg:.1f}单/天 高 {diff_pct:.1f}%；但工作日绝对量为 <strong>{weekday_count}单</strong>，仍是主要治理场景。"
        elif weekend_avg < weekday_avg:
            insight_1 = f"工作日日均 <strong>{weekday_avg:.1f}单/天</strong>，高于周末 {weekend_avg:.1f}单/天；说明 cancel 主要压力集中在工作日运营链路。"
        else:
            insight_1 = f"工作日和周末日均 cancel 基本持平，工作日 {weekday_count}单、周末 {weekend_count}单；建议同时看直播时段和原因结构。"
    else:
        insight_1 = f"当前日期区间内共 <strong>{total}单</strong> cancel；日期范围较短，日均对比建议结合更长周期观察。"

    insight_2 = f"直播时段 cancel 占 <strong>{pct(live_count, total):.1f}%</strong>，聚合小时强度约为非直播的 <strong>{intensity_multiple:.1f}倍</strong>（直播 {live_intensity:.1f}单/小时 vs 非直播 {nonlive_intensity:.1f}单/小时）。"

    if live1_count >= live2_count:
        insight_3 = f"直播①（{live1_start}:00–{live1_end}:00）贡献 <strong>{live1_count}单</strong>，高于直播②的 {live2_count}单；优先检查早/中场直播的促销说明、尺码讲解和下单引导。"
    else:
        insight_3 = f"直播②（{live2_start}:00–{live2_end}:00）贡献 <strong>{live2_count}单</strong>，高于直播①的 {live1_count}单；优先检查晚场直播的冲动下单、价格预期和支付链路。"

    insight_4 = f"Top cancel 原因为 <strong>{html.escape(top_reason)}</strong>，共 {top_reason_count}单，占 {top_reason_pct:.1f}%。如果该原因长期第一，需要把它拆到商品页、直播话术、结账页三个环节排查。"

    rec_parts = [f"优先关注 <strong>{pk_label}</strong>（峰值 {pk_val}单）的下单后反悔/取消链路"]
    if mistake_count:
        rec_parts.append(f"针对 “Bought by mistake” 类原因（{mistake_count}单）增加直播口播二次确认、规格/尺码提醒、购物车检查提示")
    if payment_count:
        rec_parts.append(f"针对支付方式变更（{payment_count}单）优化结账前支付提醒")
    if address_count:
        rec_parts.append(f"针对地址问题（{address_count}单）加强下单前地址确认")
    insight_5 = "；".join(rec_parts) + "。"

    return [insight_1, insight_2, insight_3, insight_4, insight_5]


def build_report_context(df: pd.DataFrame, raw_rows: int, unique_orders_before_filter: int,
                         source_name: str, start_date: date, end_date: date,
                         live1_start: int, live1_end: int, live2_start: int, live2_end: int):
    total = len(df)
    total_days, weekday_days, weekend_days = calendar_day_counts(start_date, end_date)

    weekday_df = df[df["Day Type"] == "工作日 Weekday"]
    weekend_df = df[df["Day Type"] == "周末 Weekend"]
    live1_df = df[df["Live Segment"] == "直播①"]
    live2_df = df[df["Live Segment"] == "直播②"]
    nonlive_df = df[df["Live Segment"] == "非直播"]

    weekday_count = len(weekday_df)
    weekend_count = len(weekend_df)
    live_count = len(live1_df) + len(live2_df)
    nonlive_count = len(nonlive_df)

    wd_hour = hourly_counts(weekday_df)
    we_hour = hourly_counts(weekend_df)
    all_hour = hourly_counts(df)

    wd_peak_label, wd_peak_val = peak_label(wd_hour)
    we_peak_label, we_peak_val = peak_label(we_hour)

    report_title = f"{start_date.strftime('%Y/%m/%d')}–{end_date.strftime('%Y/%m/%d')} 取消订单分析报告"
    header_tag = f"Cancel Order Analytics · {start_date.strftime('%Y/%m/%d')}–{end_date.strftime('%Y/%m/%d')}"
    range_label = f"{start_date.strftime('%Y/%m/%d')}–{end_date.strftime('%Y/%m/%d')}"

    ctx = {
        "total": total,
        "raw_rows": raw_rows,
        "unique_orders_before_filter": unique_orders_before_filter,
        "source_name": source_name,
        "report_title": report_title,
        "header_tag": header_tag,
        "range_label": range_label,
        "start_date": start_date,
        "end_date": end_date,
        "total_days": total_days,
        "weekday_days": weekday_days,
        "weekend_days": weekend_days,
        "weekday_count": weekday_count,
        "weekend_count": weekend_count,
        "weekday_avg": safe_div(weekday_count, weekday_days),
        "weekend_avg": safe_div(weekend_count, weekend_days),
        "live_count": live_count,
        "nonlive_count": nonlive_count,
        "live_pct": pct(live_count, total),
        "nonlive_pct": pct(nonlive_count, total),
        "wd_live_count": int(weekday_df["Is Live"].sum()) if weekday_count else 0,
        "we_live_count": int(weekend_df["Is Live"].sum()) if weekend_count else 0,
        "wd_nonlive_count": int((~weekday_df["Is Live"]).sum()) if weekday_count else 0,
        "we_nonlive_count": int((~weekend_df["Is Live"]).sum()) if weekend_count else 0,
        "wd_live_pct": pct(int(weekday_df["Is Live"].sum()) if weekday_count else 0, weekday_count),
        "we_live_pct": pct(int(weekend_df["Is Live"].sum()) if weekend_count else 0, weekend_count),
        "wd_hour": wd_hour,
        "we_hour": we_hour,
        "all_hour": all_hour,
        "wd_peak_label": wd_peak_label,
        "wd_peak_val": wd_peak_val,
        "we_peak_label": we_peak_label,
        "we_peak_val": we_peak_val,
        "weekday_reasons": reason_table(weekday_df, 7),
        "weekend_reasons": reason_table(weekend_df, 7),
        "live1_reasons": reason_table(live1_df, 5),
        "live2_reasons": reason_table(live2_df, 5),
        "nonlive_reasons": reason_table(nonlive_df, 5),
        "live1_count": len(live1_df),
        "live2_count": len(live2_df),
        "live1_start": live1_start,
        "live1_end": live1_end,
        "live2_start": live2_start,
        "live2_end": live2_end,
        "insights": make_insights(df, start_date, end_date, live1_start, live1_end, live2_start, live2_end),
    }
    return ctx


def build_html_report(ctx):
    wd_max = max(5, int(np.ceil(max(ctx["wd_hour"] + [1]) / 5.0) * 5))
    we_max = max(5, int(np.ceil(max(ctx["we_hour"] + [1]) / 2.0) * 2))
    all_max = max(5, int(np.ceil(max(ctx["all_hour"] + [1]) / 5.0) * 5))

    insight_html = "\n".join(
        f'''<div class="insight"><div class="insight-icon">// {str(i).zfill(2) if i < 5 else 'REC'}</div><div>{text}</div></div>'''
        for i, text in enumerate(ctx["insights"], start=1)
    )

    chart_colors_js = f"""
      function barColor(h){{
        if(h>={ctx['live1_start']} && h<{ctx['live1_end']}) return '#d44a1e';
        if(h>={ctx['live2_start']} && h<{ctx['live2_end']}) return '#c8840a';
        return '#2d5fa8';
      }}
    """

    html_doc = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(ctx['report_title'])}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root {{
    --bg: #f7f7f5; --surface: #ffffff; --surface2: #f0f0ed; --border: rgba(0,0,0,0.08);
    --text: #1a1a18; --text-muted: #5a5c63; --text-dim: #9a9ca3;
    --live1: #d44a1e; --live2: #c8840a; --nonlive: #2d5fa8; --accent: #d44a1e; --green: #2a9e62;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif;
    --display: Georgia, "Times New Roman", serif;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans); font-weight: 300; line-height: 1.6; min-height: 100vh; }}
  header {{ border-bottom: 1px solid var(--border); padding: 48px 60px 40px; position: relative; overflow: hidden; }}
  header::before {{ content:''; position:absolute; top:-80px; right:-80px; width:400px; height:400px; background:radial-gradient(circle, rgba(212,74,30,0.08) 0%, transparent 70%); pointer-events:none; }}
  .header-tag {{ font-family:var(--mono); font-size:11px; color:var(--accent); letter-spacing:.15em; text-transform:uppercase; margin-bottom:14px; }}
  h1 {{ font-family:var(--display); font-size:48px; font-weight:700; line-height:1.15; color:var(--text); margin-bottom:10px; }}
  .header-sub {{ font-size:14px; color:var(--text-muted); letter-spacing:.02em; }}
  .header-meta {{ position:absolute; top:48px; right:60px; text-align:right; font-family:var(--mono); font-size:11px; color:var(--text-dim); line-height:2; }}
  .header-meta strong {{ display:block; font-size:28px; color:var(--text); font-weight:500; letter-spacing:-.02em; }}
  main {{ padding:0 60px 80px; }}
  .section {{ margin-top:56px; animation:fadeUp .5s ease both; }}
  .section-label {{ font-family:var(--mono); font-size:10px; letter-spacing:.18em; text-transform:uppercase; color:var(--text-dim); margin-bottom:18px; padding-bottom:10px; border-bottom:1px solid var(--border); }}
  .section-label span {{ color:var(--accent); margin-right:8px; }}
  .stat-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:var(--border); border:1px solid var(--border); border-radius:8px; overflow:hidden; }}
  .stat {{ background:var(--surface); padding:24px 22px; position:relative; }}
  .stat::after {{ content:''; position:absolute; bottom:0; left:22px; right:22px; height:2px; border-radius:2px; }}
  .stat.live1::after {{ background:var(--live1); }} .stat.live2::after {{ background:var(--live2); }} .stat.blue::after {{ background:var(--nonlive); }} .stat.green::after {{ background:var(--green); }}
  .stat-lbl {{ font-family:var(--mono); font-size:10px; color:var(--text-muted); letter-spacing:.08em; margin-bottom:10px; }}
  .stat-val {{ font-size:36px; font-weight:500; letter-spacing:-.03em; line-height:1; margin-bottom:6px; }}
  .stat-val.orange {{ color:var(--live1); }} .stat-val.amber {{ color:var(--live2); }} .stat-val.blue {{ color:#6b9ddb; }} .stat-val.green {{ color:var(--green); }}
  .stat-sub {{ font-size:12px; color:var(--text-dim); }}
  .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .panel, .full-panel {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:24px; }}
  .panel-title {{ font-family:var(--mono); font-size:11px; color:var(--text-muted); letter-spacing:.1em; margin-bottom:18px; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
  .badge {{ background:var(--surface2); border:1px solid var(--border); border-radius:4px; padding:2px 8px; font-size:10px; color:var(--text-dim); }}
  .badge.r {{ border-color:rgba(232,87,42,.3); color:var(--live1); }} .badge.y {{ border-color:rgba(240,167,66,.3); color:var(--live2); }}
  .mini-stats {{ display:flex; gap:8px; margin-top:14px; }}
  .mini-stat {{ flex:1; background:var(--surface2); border-radius:6px; padding:10px 12px; text-align:center; }}
  .mini-stat-lbl {{ font-size:10px; color:var(--text-dim); margin-bottom:3px; }} .mini-stat-val {{ font-size:18px; font-weight:500; color:var(--text); }} .mini-stat-sub {{ font-size:10px; color:var(--text-dim); margin-top:2px; }}
  .chart-wrap {{ position:relative; width:100%; }}
  .legend {{ display:flex; gap:16px; margin-bottom:12px; flex-wrap:wrap; }}
  .legend-item {{ display:flex; align-items:center; gap:6px; font-size:11px; color:var(--text-muted); }}
  .legend-dot {{ width:10px; height:10px; border-radius:2px; flex-shrink:0; }}
  .reason-list {{ margin-top:4px; }}
  .reason-row {{ display:flex; align-items:center; gap:10px; padding:7px 0; border-bottom:1px solid var(--border); font-size:12px; }}
  .reason-row:last-child {{ border-bottom:none; }}
  .reason-name {{ flex:1; color:var(--text-muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .reason-bar-wrap {{ width:80px; height:4px; background:var(--surface2); border-radius:2px; overflow:hidden; }}
  .reason-bar {{ height:100%; border-radius:2px; }}
  .reason-cnt {{ font-family:var(--mono); font-size:11px; color:var(--text); min-width:26px; text-align:right; }}
  .reason-pct {{ font-family:var(--mono); font-size:10px; color:var(--text-dim); min-width:42px; text-align:right; }}
  .live-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }}
  .live-panel {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:20px; position:relative; overflow:hidden; }}
  .live-panel::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; }}
  .live-panel.s1::before {{ background:var(--live1); }} .live-panel.s2::before {{ background:var(--live2); }} .live-panel.s3::before {{ background:var(--nonlive); }}
  .live-title {{ font-family:var(--mono); font-size:10px; color:var(--text-dim); letter-spacing:.1em; margin-bottom:4px; }}
  .live-count {{ font-size:30px; font-weight:500; letter-spacing:-.03em; margin-bottom:14px; }}
  .live-panel.s1 .live-count {{ color:var(--live1); }} .live-panel.s2 .live-count {{ color:var(--live2); }} .live-panel.s3 .live-count {{ color:#6b9ddb; }}
  .insights {{ display:flex; flex-direction:column; gap:10px; }}
  .insight {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px 20px; font-size:13px; color:var(--text-muted); line-height:1.7; display:flex; gap:14px; align-items:flex-start; }}
  .insight-icon {{ font-family:var(--mono); font-size:10px; color:var(--accent); letter-spacing:.05em; flex-shrink:0; padding-top:3px; }}
  .insight strong {{ color:var(--text); font-weight:500; }}
  .empty-note {{ font-size:12px; color:var(--text-dim); padding:8px 0; }}
  footer {{ border-top:1px solid var(--border); margin:0 60px; padding:24px 0; font-family:var(--mono); font-size:10px; color:var(--text-dim); display:flex; justify-content:space-between; gap:20px; }}
  @keyframes fadeUp {{ from {{ opacity:0; transform:translateY(16px); }} to {{ opacity:1; transform:translateY(0); }} }}
  @media (max-width: 900px) {{ header, main {{ padding-left:24px; padding-right:24px; }} .header-meta {{ position:static; text-align:left; margin-top:20px; }} .stat-grid, .two-col, .live-grid {{ grid-template-columns:1fr; }} footer {{ margin:0 24px; flex-direction:column; }} }}
</style>
</head>
<body>
<header>
  <div class="header-tag">{html.escape(ctx['header_tag'])}</div>
  <h1>取消订单<br>分析报告</h1>
  <p class="header-sub">工作日 vs 周末 · 直播时段重合分析 · 原因拆解 · Order ID 去重口径</p>
  <div class="header-meta"><strong>{ctx['total']}</strong>去重订单总量<br>{html.escape(ctx['range_label'])}</div>
</header>

<main>
  <div class="section">
    <div class="section-label"><span>01</span>总览 Overview</div>
    <div class="stat-grid">
      <div class="stat live1"><div class="stat-lbl">工作日 Weekday</div><div class="stat-val orange">{ctx['weekday_count']}</div><div class="stat-sub">{ctx['weekday_days']}天 · 均 {ctx['weekday_avg']:.1f}单/天</div></div>
      <div class="stat green"><div class="stat-lbl">周末 Weekend</div><div class="stat-val green">{ctx['weekend_count']}</div><div class="stat-sub">{ctx['weekend_days']}天 · 均 {ctx['weekend_avg']:.1f}单/天</div></div>
      <div class="stat live2"><div class="stat-lbl">直播时段 cancel</div><div class="stat-val amber">{ctx['live_count']}</div><div class="stat-sub">占总量 {ctx['live_pct']:.1f}%</div></div>
      <div class="stat blue"><div class="stat-lbl">非直播时段 cancel</div><div class="stat-val blue">{ctx['nonlive_count']}</div><div class="stat-sub">占总量 {ctx['nonlive_pct']:.1f}%</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-label"><span>02</span>工作日 vs 周末 · 每小时 Cancel 分布</div>
    <div class="two-col">
      <div class="panel">
        <div class="panel-title">工作日 · {ctx['weekday_count']}单 <span class="badge r">直播 {ctx['wd_live_pct']:.1f}%</span></div>
        <div class="legend"><div class="legend-item"><div class="legend-dot" style="background:var(--live1)"></div>直播① {ctx['live1_start']}–{ctx['live1_end']}点</div><div class="legend-item"><div class="legend-dot" style="background:var(--live2)"></div>直播② {ctx['live2_start']}–{ctx['live2_end']}点</div><div class="legend-item"><div class="legend-dot" style="background:var(--nonlive)"></div>非直播</div></div>
        <div class="chart-wrap" style="height:180px"><canvas id="wdHour"></canvas></div>
        <div class="mini-stats"><div class="mini-stat"><div class="mini-stat-lbl">直播时段</div><div class="mini-stat-val" style="color:var(--live1)">{ctx['wd_live_count']}单</div><div class="mini-stat-sub">{ctx['wd_live_pct']:.1f}%</div></div><div class="mini-stat"><div class="mini-stat-lbl">非直播</div><div class="mini-stat-val">{ctx['wd_nonlive_count']}单</div><div class="mini-stat-sub">{pct(ctx['wd_nonlive_count'], ctx['weekday_count']):.1f}%</div></div><div class="mini-stat"><div class="mini-stat-lbl">峰值时段</div><div class="mini-stat-val" style="color:var(--live1)">{ctx['wd_peak_label']}</div><div class="mini-stat-sub">{ctx['wd_peak_val']}单</div></div></div>
      </div>
      <div class="panel">
        <div class="panel-title">周末 · {ctx['weekend_count']}单 <span class="badge y">直播 {ctx['we_live_pct']:.1f}%</span></div>
        <div class="legend"><div class="legend-item"><div class="legend-dot" style="background:var(--live1)"></div>直播① {ctx['live1_start']}–{ctx['live1_end']}点</div><div class="legend-item"><div class="legend-dot" style="background:var(--live2)"></div>直播② {ctx['live2_start']}–{ctx['live2_end']}点</div><div class="legend-item"><div class="legend-dot" style="background:var(--nonlive)"></div>非直播</div></div>
        <div class="chart-wrap" style="height:180px"><canvas id="weHour"></canvas></div>
        <div class="mini-stats"><div class="mini-stat"><div class="mini-stat-lbl">直播时段</div><div class="mini-stat-val" style="color:var(--live1)">{ctx['we_live_count']}单</div><div class="mini-stat-sub">{ctx['we_live_pct']:.1f}%</div></div><div class="mini-stat"><div class="mini-stat-lbl">非直播</div><div class="mini-stat-val">{ctx['we_nonlive_count']}单</div><div class="mini-stat-sub">{pct(ctx['we_nonlive_count'], ctx['weekend_count']):.1f}%</div></div><div class="mini-stat"><div class="mini-stat-lbl">峰值时段</div><div class="mini-stat-val" style="color:var(--live1)">{ctx['we_peak_label']}</div><div class="mini-stat-sub">{ctx['we_peak_val']}单</div></div></div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-label"><span>03</span>取消原因对比 · 工作日 vs 周末</div>
    <div class="two-col">
      <div class="panel"><div class="panel-title">工作日 Top 原因</div><div class="reason-list">{make_reason_rows(ctx['weekday_reasons'], 'var(--live1)')}</div></div>
      <div class="panel"><div class="panel-title">周末 Top 原因</div><div class="reason-list">{make_reason_rows(ctx['weekend_reasons'], 'var(--live2)')}</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-label"><span>04</span>直播时段 Cancel 深度拆分</div>
    <div class="live-grid">
      <div class="live-panel s1"><div class="live-title">直播① · {ctx['live1_start']}:00–{ctx['live1_end']}:00</div><div class="live-count">{ctx['live1_count']}单</div><div class="reason-list">{make_reason_rows(ctx['live1_reasons'], 'var(--live1)')}</div></div>
      <div class="live-panel s2"><div class="live-title">直播② · {ctx['live2_start']}:00–{ctx['live2_end']}:00</div><div class="live-count">{ctx['live2_count']}单</div><div class="reason-list">{make_reason_rows(ctx['live2_reasons'], 'var(--live2)')}</div></div>
      <div class="live-panel s3"><div class="live-title">非直播 · 其余时段</div><div class="live-count">{ctx['nonlive_count']}单</div><div class="reason-list">{make_reason_rows(ctx['nonlive_reasons'], 'var(--nonlive)')}</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-label"><span>05</span>全天 Cancel 时段分布（所有订单合并）</div>
    <div class="full-panel"><div class="legend" style="margin-bottom:16px"><div class="legend-item"><div class="legend-dot" style="background:var(--live1)"></div>直播① {ctx['live1_start']}–{ctx['live1_end']}点</div><div class="legend-item"><div class="legend-dot" style="background:var(--live2)"></div>直播② {ctx['live2_start']}–{ctx['live2_end']}点</div><div class="legend-item"><div class="legend-dot" style="background:var(--nonlive)"></div>非直播时段</div></div><div class="chart-wrap" style="height:220px"><canvas id="allHour"></canvas></div></div>
  </div>

  <div class="section">
    <div class="section-label"><span>06</span>关键洞察 Key Insights</div>
    <div class="insights">{insight_html}</div>
  </div>
</main>

<footer>
  <span>数据来源：{html.escape(ctx['source_name'])} · 原始 {ctx['raw_rows']} 行 SKU 明细 · 去重后 {ctx['unique_orders_before_filter']} 个 Order ID</span>
  <span>直播时段定义：{ctx['live1_start']}:00–{ctx['live1_end']}:00 / {ctx['live2_start']}:00–{ctx['live2_end']}:00；去重口径：Order ID 唯一</span>
</footer>

<script>
const C = Chart;
const hrs = Array.from({{length:24}},(_,i)=>i);
{chart_colors_js}
const colors = hrs.map(barColor);
const commonOpts = (max, step) => ({{responsive:true, maintainAspectRatio:false, plugins:{{legend:{{display:false}}}}, scales:{{x:{{ticks:{{autoSkip:false,maxRotation:0,font:{{size:9}},color:'#9a9ca3'}},grid:{{color:'rgba(0,0,0,0.05)'}},border:{{color:'rgba(0,0,0,0.08)'}}}}, y:{{beginAtZero:true, max:max, ticks:{{stepSize:step,font:{{size:10}},color:'#9a9ca3'}}, grid:{{color:'rgba(0,0,0,0.05)'}}, border:{{color:'rgba(0,0,0,0.08)'}}}}}}}});
new C(document.getElementById('wdHour'), {{type:'bar', data:{{labels:hrs.map(h=>h+'h'), datasets:[{{data:{json.dumps(ctx['wd_hour'])}, backgroundColor:colors, borderRadius:3}}]}}, options:commonOpts({wd_max}, {max(1, wd_max//5)})}});
new C(document.getElementById('weHour'), {{type:'bar', data:{{labels:hrs.map(h=>h+'h'), datasets:[{{data:{json.dumps(ctx['we_hour'])}, backgroundColor:colors, borderRadius:3}}]}}, options:commonOpts({we_max}, {max(1, we_max//5)})}});
new C(document.getElementById('allHour'), {{type:'bar', data:{{labels:hrs.map(h=>h+'h'), datasets:[{{data:{json.dumps(ctx['all_hour'])}, backgroundColor:colors, borderRadius:3}}]}}, options:commonOpts({all_max}, {max(1, all_max//5)})}});
</script>
</body>
</html>'''
    return html_doc


def make_excel_download(order_df: pd.DataFrame, analysis_df: pd.DataFrame, ctx: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary = pd.DataFrame([
            ["原始 SKU 行数", ctx["raw_rows"]],
            ["去重 Order ID 数", ctx["unique_orders_before_filter"]],
            ["当前日期区间 cancel 单数", ctx["total"]],
            ["工作日 cancel", ctx["weekday_count"]],
            ["周末 cancel", ctx["weekend_count"]],
            ["直播时段 cancel", ctx["live_count"]],
            ["非直播时段 cancel", ctx["nonlive_count"]],
            ["直播时段占比", f"{ctx['live_pct']:.1f}%"],
        ], columns=["Metric", "Value"])
        summary.to_excel(writer, index=False, sheet_name="Summary")

        reason_all = analysis_df["Cancel Reason Clean"].value_counts().reset_index()
        reason_all.columns = ["Cancel Reason", "Order Count"]
        reason_all["Share"] = reason_all["Order Count"] / max(1, len(analysis_df))
        reason_all.to_excel(writer, index=False, sheet_name="Reason Breakdown")

        hour_all = pd.DataFrame({
            "Hour": list(range(24)),
            "All Orders": ctx["all_hour"],
            "Weekday": ctx["wd_hour"],
            "Weekend": ctx["we_hour"],
        })
        hour_all.to_excel(writer, index=False, sheet_name="Hourly Breakdown")

        export_cols = [c for c in ["Order ID", "Cancelled Datetime", "Created Datetime", "Cancel Reason", "Cancel Reason Clean", "Day Type", "Live Segment", "Hour", "SKU Count", "Item Quantity", "Seller SKU", "Variation", "Order Amount Parsed", "Order Refund Amount Parsed", "Payment Method", "State", "Country"] if c in analysis_df.columns]
        analysis_df[export_cols].to_excel(writer, index=False, sheet_name="Cleaned Orders")
    return output.getvalue()


# =========================
# UI
# =========================
st.title("📉 Cancel Order Report Generator")
st.caption("上传 TikTok Shop cancelled orders 明细表后，自动按 Order ID 去重，并生成与示例附件同结构的取消订单分析报告。")

uploaded_file = st.file_uploader("上传 Cancelled Orders 详情表（CSV / Excel）", type=["csv", "xlsx", "xls"])

with st.sidebar:
    st.header("报告设置")
    st.caption("默认口径：一个 Order ID = 一个 cancel。")
    live1_start = st.number_input("直播①开始小时", min_value=0, max_value=23, value=10, step=1)
    live1_end = st.number_input("直播①结束小时（不含）", min_value=1, max_value=24, value=18, step=1)
    live2_start = st.number_input("直播②开始小时", min_value=0, max_value=23, value=19, step=1)
    live2_end = st.number_input("直播②结束小时（不含）", min_value=1, max_value=24, value=23, step=1)
    top_n = st.slider("页面表格展示 Top N 原因", 3, 12, 7, 1)
    show_cleaned = st.checkbox("显示订单级明细预览", value=True)

if uploaded_file is None:
    st.info("请先上传一份 TikTok Shop cancelled orders 详情表。")
    st.markdown("""
**程序会自动完成：**
1. 读取未清洗表格；
2. 以 `Order ID` 去重，把一行一个 SKU 的明细聚合为一行一个订单；
3. 使用 `Cancelled Time` 判断日期、小时、工作日/周末和直播时段；
4. 使用 `Cancel Reason` 统计原因占比；
5. 生成 HTML 报告、清洗后的订单级 Excel。
""")
    st.stop()

try:
    raw = read_uploaded_file(uploaded_file)
    raw.columns = [clean_column_name(c) for c in raw.columns]
except Exception as e:
    st.error(f"文件读取失败：{e}")
    st.stop()

st.subheader("1）字段识别")
cols = raw.columns.tolist()

if "Order ID" not in cols:
    st.error("没有识别到 `Order ID` 字段。请确认上传的是 TikTok Shop cancelled orders 明细表。")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    order_id_col = st.selectbox("Order ID 字段", cols, index=cols.index("Order ID"))
with col2:
    default_time = "Cancelled Time" if "Cancelled Time" in cols else next((c for c in cols if "Time" in c), cols[0])
    time_col_raw = st.selectbox("用于分析的时间字段", cols, index=cols.index(default_time))
with col3:
    default_reason = "Cancel Reason" if "Cancel Reason" in cols else next((c for c in cols if "Reason" in c), cols[0])
    reason_col = st.selectbox("取消原因字段", cols, index=cols.index(default_reason))

order_df = build_order_level(raw, order_id_col)

# Normalize the selected time column to the order-level parsed column if possible.
parsed_candidate = f"__parsed_{time_col_raw}"
if time_col_raw == "Cancelled Time" and "Cancelled Datetime" in order_df.columns:
    time_col = "Cancelled Datetime"
elif time_col_raw == "Created Time" and "Created Datetime" in order_df.columns:
    time_col = "Created Datetime"
elif time_col_raw == "Paid Time" and "Paid Datetime" in order_df.columns:
    time_col = "Paid Datetime"
elif parsed_candidate in order_df.columns:
    order_df[time_col_raw + " Parsed"] = order_df[parsed_candidate]
    time_col = time_col_raw + " Parsed"
else:
    order_df[time_col_raw + " Parsed"] = parse_datetime_series(order_df[time_col_raw]) if time_col_raw in order_df.columns else pd.NaT
    time_col = time_col_raw + " Parsed"

if reason_col not in order_df.columns:
    # If grouped under the same name was not carried through for some reason, create Unknown.
    order_df[reason_col] = "Unknown"

valid_dates = order_df[time_col].dropna()
if len(valid_dates) == 0:
    st.error(f"`{time_col_raw}` 无法解析出有效日期，请换一个时间字段或检查表格格式。")
    st.stop()

min_date = valid_dates.dt.date.min()
max_date = valid_dates.dt.date.max()

st.subheader("2）日期区间")
st.caption("如果导出的周/月区间首尾没有 cancel，建议手动把日期改成完整周/月区间；否则日均会按文件中最早/最晚 cancel 日期计算。")
selected_range = st.date_input(
    "选择报告统计日期区间（默认使用文件里的最小/最大取消日期）",
    value=(min_date, max_date),
)
if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date, end_date = min_date, max_date

if start_date > end_date:
    st.error("开始日期不能晚于结束日期。")
    st.stop()

if not (0 <= live1_start < live1_end <= 24 and 0 <= live2_start < live2_end <= 24):
    st.error("直播时段设置不合法：开始小时必须小于结束小时，且范围在 0–24 之间。")
    st.stop()

analysis_df = prepare_analysis(order_df, time_col, reason_col, start_date, end_date, live1_start, live1_end, live2_start, live2_end)

# Recompute reason tables with selected top_n after context build.
ctx = build_report_context(
    analysis_df,
    raw_rows=len(raw),
    unique_orders_before_filter=order_df["Order ID"].nunique(),
    source_name=uploaded_file.name,
    start_date=start_date,
    end_date=end_date,
    live1_start=int(live1_start), live1_end=int(live1_end),
    live2_start=int(live2_start), live2_end=int(live2_end),
)
ctx["weekday_reasons"] = reason_table(analysis_df[analysis_df["Day Type"] == "工作日 Weekday"], top_n)
ctx["weekend_reasons"] = reason_table(analysis_df[analysis_df["Day Type"] == "周末 Weekend"], top_n)
ctx["live1_reasons"] = reason_table(analysis_df[analysis_df["Live Segment"] == "直播①"], min(top_n, 7))
ctx["live2_reasons"] = reason_table(analysis_df[analysis_df["Live Segment"] == "直播②"], min(top_n, 7))
ctx["nonlive_reasons"] = reason_table(analysis_df[analysis_df["Live Segment"] == "非直播"], min(top_n, 7))

if len(analysis_df) == 0:
    st.warning("当前日期区间内没有可分析的 cancel 订单。")
    st.stop()

st.subheader("3）核心结果")
a, b, c, d = st.columns(4)
a.metric("原始 SKU 行数", f"{len(raw):,}")
b.metric("去重 Order ID", f"{order_df['Order ID'].nunique():,}")
c.metric("当前区间 Cancel", f"{len(analysis_df):,}")
d.metric("直播时段占比", f"{ctx['live_pct']:.1f}%")

html_report = build_html_report(ctx)
excel_bytes = make_excel_download(order_df, analysis_df, ctx)
file_stub = f"cancel_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

st.download_button(
    "⬇️ 下载 HTML Report",
    data=html_report.encode("utf-8"),
    file_name=f"{file_stub}.html",
    mime="text/html",
)
st.download_button(
    "⬇️ 下载清洗后的订单级 Excel",
    data=excel_bytes,
    file_name=f"{file_stub}_cleaned_orders.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.subheader("4）报告预览")
st.caption("下面是最终 HTML 报告预览；下载 HTML 后可以直接发给团队或截图放进周报/月报。")
components.html(html_report, height=2300, scrolling=True)

with st.expander("查看自动生成的关键洞察"):
    for i, insight in enumerate(ctx["insights"], start=1):
        st.markdown(f"**{i}.** {insight}", unsafe_allow_html=True)

if show_cleaned:
    st.subheader("5）订单级明细预览")
    preview_cols = [c for c in ["Order ID", "Cancelled Datetime", "Cancel Reason", "Cancel Reason Clean", "Day Type", "Live Segment", "Hour", "SKU Count", "Item Quantity", "Seller SKU", "Variation", "Order Amount Parsed", "Order Refund Amount Parsed", "Payment Method", "State", "Country"] if c in analysis_df.columns]
    st.dataframe(analysis_df[preview_cols].sort_values("Cancelled Datetime" if "Cancelled Datetime" in preview_cols else preview_cols[0]), use_container_width=True, height=360)

st.caption("说明：直播时段只是“取消时间与直播时段重合”的分析，不等同于因果归因。建议结合直播间 GMV、订单量、曝光、主播话术和活动价变动一起判断。")
