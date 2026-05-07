import html
import json
from io import BytesIO
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Cancelled Orders Report Generator",
    page_icon="💅",
    layout="wide",
)


# =========================
# Constants
# =========================
REQUIRED_COLUMNS = ["Order ID", "Order Status", "Created Time", "Product Name"]
OPTIONAL_COLUMNS = [
    "Cancelled Time", "Cancel Reason", "Cancel By", "Seller SKU", "SKU ID",
    "Variation", "Quantity", "Order Amount", "SKU Subtotal After Discount",
    "Cancelation/Return Type", "Payment Method", "Fulfillment Type", "Delivery Option",
]
SIZE_TOKENS = {
    "XS", "S", "M", "L", "XL", "XXL", "XXXL",
    "EXTRA SMALL", "SMALL", "MEDIUM", "LARGE", "EXTRA LARGE",
}


# =========================
# Basic helpers
# =========================
def clean_column_name(col: str) -> str:
    return str(col).replace("\ufeff", "").strip()


def stringify_id(x) -> str:
    """Keep long TikTok IDs safe as strings, without .0 / tabs / spaces."""
    if pd.isna(x):
        return ""
    s = str(x).replace("\t", "").replace("\r", "").replace("\n", "").strip()
    if s.endswith(".0") and s[:-2].replace(".", "", 1).isdigit():
        s = s[:-2]
    return s


def normalize_text(x, default="Unknown") -> str:
    if pd.isna(x):
        return default
    s = str(x).replace("\t", "").replace("\r", " ").replace("\n", " ").strip()
    if not s or s.lower() in {"nan", "nat", "none"}:
        return default
    return s


def parse_datetime_series(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
        .str.replace("\t", "", regex=False)
        .str.replace("\r", "", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "NaT": np.nan, "None": np.nan})
    )
    # TikTok commonly exports: 04/05/2026 11:54:28 PM
    parsed = pd.to_datetime(cleaned, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    missing = parsed.isna() & cleaned.notna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(cleaned.loc[missing], errors="coerce")
    return parsed


def parse_number_series(s: pd.Series, default=np.nan) -> pd.Series:
    out = pd.to_numeric(
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("\t", "", regex=False)
        .str.strip(),
        errors="coerce",
    )
    if not pd.isna(default):
        out = out.fillna(default)
    return out


def pct(n, d, digits=1):
    if d == 0 or pd.isna(d):
        return 0.0
    return round(float(n) / float(d) * 100, digits)


def safe_div(n, d):
    return float(n) / float(d) if d else 0.0


def fmt_pct(n, d, digits=1):
    return f"{pct(n, d, digits):.{digits}f}%"


def fmt_num(x, digits=0):
    if pd.isna(x):
        x = 0
    if digits == 0:
        return f"{int(round(float(x))):,}"
    return f"{float(x):,.{digits}f}"


def fmt_money(x):
    if pd.isna(x):
        x = 0
    return f"${float(x):,.2f}"


def mode_or_first(series: pd.Series, default="Unknown"):
    x = series.dropna().astype(str).str.replace("\t", "", regex=False).str.strip()
    x = x[(x != "") & (~x.str.lower().isin(["nan", "nat", "none"]))]
    if len(x) == 0:
        return default
    mode = x.mode()
    return mode.iloc[0] if len(mode) else x.iloc[0]


def first_non_null(series: pd.Series):
    x = series.dropna()
    x = x[x.astype(str).str.strip() != ""]
    return x.iloc[0] if len(x) else np.nan


def join_unique(series: pd.Series, max_items=50):
    vals, seen = [], set()
    for v in series.dropna().astype(str):
        s = normalize_text(v, default="")
        if s and s not in seen:
            seen.add(s)
            vals.append(s)
        if len(vals) >= max_items:
            vals.append("...")
            break
    return "; ".join(vals)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
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


def derive_nail_style(row) -> str:
    """
    TikTok Variation is usually like 'Acai Bloom, L'.
    For nail-style breakdown, remove the final size token when possible.
    Fallback: Seller SKU, then Product Name.
    """
    variation = normalize_text(row.get("Variation", ""), default="")
    if variation:
        parts = [p.strip() for p in variation.split(",") if p.strip()]
        if len(parts) >= 2 and parts[-1].upper() in SIZE_TOKENS:
            return ", ".join(parts[:-1]).strip() or variation
        return variation
    sku = normalize_text(row.get("Seller SKU", ""), default="")
    if sku:
        return sku
    return normalize_text(row.get("Product Name", ""), default="Unknown")


def classify_live_segment(hour, live1_start, live1_end, live2_start, live2_end) -> str:
    if pd.isna(hour):
        return "Unknown"
    h = int(hour)
    if live1_start <= h < live1_end:
        return "直播①"
    if live2_start <= h < live2_end:
        return "直播②"
    return "非直播"


def calendar_day_counts(start_date: date, end_date: date):
    days = pd.date_range(start=start_date, end=end_date, freq="D")
    weekday_days = int((days.weekday < 5).sum())
    weekend_days = int((days.weekday >= 5).sum())
    return len(days), weekday_days, weekend_days


def hourly_counts_from_series(dt_series: pd.Series):
    valid = dt_series.dropna()
    if len(valid) == 0:
        return [0] * 24
    s = valid.dt.hour.value_counts()
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


# =========================
# Cleaning + aggregation
# =========================
def validate_columns(df: pd.DataFrame):
    df.columns = [clean_column_name(c) for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


def prepare_line_level(raw: pd.DataFrame, metric_mode: str) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    df["Order ID"] = df["Order ID"].apply(stringify_id)
    df = df[df["Order ID"] != ""].copy()

    df["Order Status Clean"] = df["Order Status"].apply(lambda x: normalize_text(x, default="Unknown"))
    df["Is Cancelled Line"] = df["Order Status Clean"].str.lower().isin(["cancelled", "canceled"])

    df["Created Datetime"] = parse_datetime_series(df["Created Time"])
    if "Cancelled Time" in df.columns:
        df["Cancelled Datetime"] = parse_datetime_series(df["Cancelled Time"])
    else:
        df["Cancelled Datetime"] = pd.NaT

    if "Cancel Reason" in df.columns:
        df["Cancel Reason Clean"] = df["Cancel Reason"].apply(lambda x: normalize_text(x, default="Unknown"))
    else:
        df["Cancel Reason Clean"] = "Unknown"

    if "Product Name" in df.columns:
        df["Product Link / Product Name"] = df["Product Name"].apply(lambda x: normalize_text(x, default="Unknown"))
    else:
        df["Product Link / Product Name"] = "Unknown"

    if "Variation" not in df.columns:
        df["Variation"] = ""
    if "Seller SKU" not in df.columns:
        df["Seller SKU"] = ""

    df["Nail Style"] = df.apply(derive_nail_style, axis=1)

    if "Quantity" in df.columns:
        qty = parse_number_series(df["Quantity"], default=np.nan)
        qty = qty.fillna(1)
        qty = qty.where(qty > 0, 1)
    else:
        qty = pd.Series([1] * len(df), index=df.index)

    df["SKU Row Count"] = 1
    df["Quantity Parsed"] = qty
    if metric_mode == "quantity":
        df["Metric Units"] = df["Quantity Parsed"]
        df["Metric Label"] = "按 Quantity 汇总"
    else:
        df["Metric Units"] = 1
        df["Metric Label"] = "按 SKU 行数汇总"

    for c in ["Order Amount", "SKU Subtotal After Discount", "Order Refund Amount"]:
        if c in df.columns:
            df[f"{c} Parsed"] = parse_number_series(df[c], default=0)

    return df


def filter_by_created_date(lines: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    df = lines.copy()
    df = df[df["Created Datetime"].notna()].copy()
    df["Created Date"] = df["Created Datetime"].dt.date
    return df[(df["Created Date"] >= start_date) & (df["Created Date"] <= end_date)].copy()


def build_order_level(lines: pd.DataFrame, live1_start, live1_end, live2_start, live2_end) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame()

    tmp = lines.copy()
    tmp["__status_cancelled_int"] = tmp["Is Cancelled Line"].astype(int)

    agg = {
        "Created Datetime": "min",
        "Cancelled Datetime": first_non_null,
        "Order Status Clean": mode_or_first,
        "__status_cancelled_int": "max",
        "Cancel Reason Clean": mode_or_first,
        "Nail Style": join_unique,
        "Product Link / Product Name": join_unique,
        "Metric Units": "sum",
        "SKU Row Count": "sum",
        "Quantity Parsed": "sum",
    }
    if "Seller SKU" in tmp.columns:
        agg["Seller SKU"] = join_unique
    if "Variation" in tmp.columns:
        agg["Variation"] = join_unique
    if "Order Amount Parsed" in tmp.columns:
        agg["Order Amount Parsed"] = first_non_null
    if "SKU Subtotal After Discount Parsed" in tmp.columns:
        agg["SKU Subtotal After Discount Parsed"] = "sum"

    od = tmp.groupby("Order ID", as_index=False).agg(agg)
    od["Is Cancelled"] = od["__status_cancelled_int"].eq(1)
    od["Order Status Final"] = np.where(od["Is Cancelled"], "Cancelled", od["Order Status Clean"])
    od["Created Hour"] = od["Created Datetime"].dt.hour
    od["Created Weekday Num"] = od["Created Datetime"].dt.weekday
    od["Created Day Type"] = np.where(od["Created Weekday Num"] >= 5, "周末 Weekend", "工作日 Weekday")
    od["Live Segment by Created Time"] = od["Created Hour"].apply(
        lambda h: classify_live_segment(h, live1_start, live1_end, live2_start, live2_end)
    )
    od["Is Live by Created Time"] = od["Live Segment by Created Time"].isin(["直播①", "直播②"])
    od["Cancelled Hour"] = od["Cancelled Datetime"].dt.hour
    od["Cancelled Day Type"] = np.where(
        od["Cancelled Datetime"].dt.weekday >= 5, "周末 Weekend", "工作日 Weekday"
    )
    return od.drop(columns=["__status_cancelled_int"])


def make_breakdown(lines: pd.DataFrame, group_col: str, top_n: int, metric_label: str) -> pd.DataFrame:
    if lines.empty or group_col not in lines.columns:
        return pd.DataFrame(columns=[group_col, metric_label, "Order Count", "Pct", "SKU Row Count", "Quantity"])
    total_metric = lines["Metric Units"].sum()
    out = (
        lines.groupby(group_col, dropna=False)
        .agg(
            **{
                metric_label: ("Metric Units", "sum"),
                "Order Count": ("Order ID", "nunique"),
                "SKU Row Count": ("SKU Row Count", "sum"),
                "Quantity": ("Quantity Parsed", "sum"),
            }
        )
        .reset_index()
    )
    out[group_col] = out[group_col].apply(lambda x: normalize_text(x, default="Unknown"))
    out["Pct"] = out[metric_label].apply(lambda x: pct(x, total_metric))
    out = out.sort_values([metric_label, "Order Count"], ascending=[False, False]).head(top_n).reset_index(drop=True)
    return out


def value_count_table(df: pd.DataFrame, col: str, denominator: int, top_n=10) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[col, "Order Count", "Pct"])
    out = df[col].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"}).value_counts().head(top_n).reset_index()
    out.columns = [col, "Order Count"]
    out["Pct"] = out["Order Count"].apply(lambda x: pct(x, denominator))
    return out


# =========================
# HTML rendering helpers
# =========================
def make_reason_rows(reason_df: pd.DataFrame, color="var(--live1)"):
    if reason_df.empty:
        return '<div class="empty-note">暂无数据</div>'
    max_count = reason_df["Order Count"].max()
    rows = []
    for _, r in reason_df.iterrows():
        name = str(r.iloc[0])
        width = pct(r["Order Count"], max_count, 0) if max_count else 0
        rows.append(f'''
          <div class="reason-row">
            <div class="reason-name" title="{html.escape(name)}">{html.escape(name)}</div>
            <div class="reason-bar-wrap"><div class="reason-bar" style="width:{width}%;background:{color}"></div></div>
            <div class="reason-cnt">{fmt_num(r['Order Count'])}</div>
            <div class="reason-pct">{float(r['Pct']):.1f}%</div>
          </div>
        ''')
    return "\n".join(rows)


def make_breakdown_rows(bdf: pd.DataFrame, name_col: str, metric_col: str, color="var(--live1)"):
    if bdf.empty:
        return '<div class="empty-note">暂无数据</div>'
    max_metric = bdf[metric_col].max()
    rows = []
    for _, r in bdf.iterrows():
        name = str(r[name_col])
        width = pct(r[metric_col], max_metric, 0) if max_metric else 0
        rows.append(f'''
          <div class="break-row">
            <div class="break-name" title="{html.escape(name)}">{html.escape(name)}</div>
            <div class="break-bar-wrap"><div class="break-bar" style="width:{width}%;background:{color}"></div></div>
            <div class="break-num">{fmt_num(r[metric_col])}</div>
            <div class="break-orders">{fmt_num(r['Order Count'])} orders</div>
            <div class="break-pct">{float(r['Pct']):.1f}%</div>
          </div>
        ''')
    return "\n".join(rows)


def build_insights(all_orders: pd.DataFrame, cancel_orders: pd.DataFrame, cancel_lines: pd.DataFrame,
                   style_breakdown: pd.DataFrame, product_breakdown: pd.DataFrame,
                   start_date: date, end_date: date, live1_start, live1_end, live2_start, live2_end):
    total_orders = len(all_orders)
    cancel_orders_n = len(cancel_orders)
    cancel_rate = pct(cancel_orders_n, total_orders)

    live_all = int(all_orders["Is Live by Created Time"].sum()) if not all_orders.empty else 0
    nonlive_all = total_orders - live_all
    live_cancel = int(cancel_orders["Is Live by Created Time"].sum()) if not cancel_orders.empty else 0
    nonlive_cancel = cancel_orders_n - live_cancel
    live_cancel_rate = pct(live_cancel, live_all)
    nonlive_cancel_rate = pct(nonlive_cancel, nonlive_all)

    created_counts = hourly_counts_from_series(cancel_orders["Created Datetime"] if not cancel_orders.empty else pd.Series(dtype="datetime64[ns]"))
    cancel_counts = hourly_counts_from_series(cancel_orders["Cancelled Datetime"] if not cancel_orders.empty else pd.Series(dtype="datetime64[ns]"))
    created_peak_label, created_peak_val = peak_label(created_counts)
    cancel_peak_label, cancel_peak_val = peak_label(cancel_counts)

    reason_df = value_count_table(cancel_orders, "Cancel Reason Clean", cancel_orders_n, top_n=5)
    top_reason = str(reason_df.iloc[0, 0]) if not reason_df.empty else "-"
    top_reason_count = int(reason_df.iloc[0]["Order Count"]) if not reason_df.empty else 0
    top_reason_pct = float(reason_df.iloc[0]["Pct"]) if not reason_df.empty else 0

    top_style = "-"
    top_style_pct = 0
    if not style_breakdown.empty:
        top_style = str(style_breakdown.iloc[0]["Nail Style"])
        top_style_pct = float(style_breakdown.iloc[0]["Pct"])

    top_product = "-"
    top_product_pct = 0
    if not product_breakdown.empty:
        top_product = str(product_breakdown.iloc[0]["Product Link / Product Name"])
        top_product_pct = float(product_breakdown.iloc[0]["Pct"])

    weekday_all = int((all_orders["Created Day Type"] == "工作日 Weekday").sum()) if not all_orders.empty else 0
    weekend_all = total_orders - weekday_all
    weekday_cancel = int((cancel_orders["Created Day Type"] == "工作日 Weekday").sum()) if not cancel_orders.empty else 0
    weekend_cancel = cancel_orders_n - weekday_cancel

    insights = []
    insights.append(
        f"本周期总订单 <strong>{fmt_num(total_orders)}</strong> 单，其中 Cancelled <strong>{fmt_num(cancel_orders_n)}</strong> 单，订单级 cancel 占比为 <strong>{cancel_rate:.1f}%</strong>。该口径已按 Order ID 去重，不受一个订单多个 SKU 行影响。"
    )

    if live_all or nonlive_all:
        if live_cancel_rate > nonlive_cancel_rate:
            diff = live_cancel_rate - nonlive_cancel_rate
            insights.append(
                f"按 <strong>Created Time</strong> 归因，直播时段创建订单的 cancel rate 为 <strong>{live_cancel_rate:.1f}%</strong>，非直播为 {nonlive_cancel_rate:.1f}%，直播购买链路高出 {diff:.1f} 个百分点。建议重点检查直播口播、价格预期、尺码解释和冲动下单后的反悔。"
            )
        else:
            insights.append(
                f"按 <strong>Created Time</strong> 归因，直播时段创建订单的 cancel rate 为 <strong>{live_cancel_rate:.1f}%</strong>，非直播为 {nonlive_cancel_rate:.1f}%。当前数据不显示直播购买本身比非直播更容易取消，需继续看具体原因和产品链接。"
            )

    insights.append(
        f"Cancelled orders 的创建高峰在 <strong>{created_peak_label}</strong>（{fmt_num(created_peak_val)}单），实际取消动作高峰在 <strong>{cancel_peak_label}</strong>（{fmt_num(cancel_peak_val)}单）。Created Time 用于直播归因，Cancelled Time 仅用于判断顾客什么时候发起/完成取消。"
    )

    if top_reason != "-":
        insights.append(
            f"Top cancel reason 是 <strong>{html.escape(top_reason)}</strong>，共 {fmt_num(top_reason_count)} 单，占 {top_reason_pct:.1f}%。如果该原因连续多周第一，建议把它拆到直播话术、PDP 信息、结账页、客服拦截四个环节排查。"
        )

    if top_style != "-":
        insights.append(
            f"Cancelled SKU/甲型中占比最高的是 <strong>{html.escape(top_style)}</strong>，占 SKU 维度 {top_style_pct:.1f}%。建议结合该甲型的尺码选择、主图预期、价格折扣和是否直播强推一起看。"
        )

    if top_product != "-":
        insights.append(
            f"H column 产品链接维度中，取消占比最高的是 <strong>{html.escape(top_product)}</strong>，占 SKU 维度 {top_product_pct:.1f}%。如果该链接长期集中，优先检查该链接的标题、图片、变体、优惠展示和库存/尺码选择。"
        )

    if weekday_all and weekend_all:
        insights.append(
            f"工作日 Created Time 订单 cancel rate 为 <strong>{pct(weekday_cancel, weekday_all):.1f}%</strong>，周末为 <strong>{pct(weekend_cancel, weekend_all):.1f}%</strong>。建议每周固定对比，判断问题更偏向工作日直播流量质量，还是周末冲动购买。"
        )

    return insights[:7]


def build_html_report(ctx):
    created_max = max(5, int(np.ceil(max(ctx["created_hour_counts"] + [1]) / 5.0) * 5))
    cancelled_max = max(5, int(np.ceil(max(ctx["cancelled_hour_counts"] + [1]) / 5.0) * 5))

    insights_html = "\n".join(
        f'''<div class="insight"><div class="insight-icon">// {str(i).zfill(2) if i < len(ctx['insights']) else 'REC'}</div><div>{txt}</div></div>'''
        for i, txt in enumerate(ctx["insights"], start=1)
    )

    html_doc = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(ctx['report_title'])}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root {{
    --bg:#f7f7f5; --surface:#ffffff; --surface2:#f0f0ed; --border:rgba(0,0,0,0.08);
    --text:#1a1a18; --text-muted:#5a5c63; --text-dim:#9a9ca3;
    --live1:#d44a1e; --live2:#c8840a; --nonlive:#2d5fa8; --accent:#d44a1e; --green:#2a9e62; --purple:#7a4cc2;
    --mono:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC","Microsoft YaHei",Arial,sans-serif;
    --display:Georgia,"Times New Roman",serif;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:var(--sans); font-weight:300; line-height:1.6; min-height:100vh; }}
  header {{ border-bottom:1px solid var(--border); padding:48px 60px 40px; position:relative; overflow:hidden; }}
  header::before {{ content:''; position:absolute; top:-80px; right:-80px; width:400px; height:400px; background:radial-gradient(circle,rgba(212,74,30,0.08) 0%,transparent 70%); pointer-events:none; }}
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
  .stat.red::after {{ background:var(--live1); }} .stat.green::after {{ background:var(--green); }} .stat.blue::after {{ background:var(--nonlive); }} .stat.amber::after {{ background:var(--live2); }} .stat.purple::after {{ background:var(--purple); }}
  .stat-lbl {{ font-family:var(--mono); font-size:10px; color:var(--text-muted); letter-spacing:.08em; margin-bottom:10px; }}
  .stat-val {{ font-size:34px; font-weight:500; letter-spacing:-.03em; line-height:1; margin-bottom:6px; }}
  .stat-val.red {{ color:var(--live1); }} .stat-val.green {{ color:var(--green); }} .stat-val.blue {{ color:#6b9ddb; }} .stat-val.amber {{ color:var(--live2); }} .stat-val.purple {{ color:var(--purple); }}
  .stat-sub {{ font-size:12px; color:var(--text-dim); }}
  .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .three-col {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }}
  .panel,.full-panel {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:24px; }}
  .panel-title {{ font-family:var(--mono); font-size:11px; color:var(--text-muted); letter-spacing:.1em; margin-bottom:18px; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
  .badge {{ background:var(--surface2); border:1px solid var(--border); border-radius:4px; padding:2px 8px; font-size:10px; color:var(--text-dim); }}
  .badge.r {{ border-color:rgba(232,87,42,.3); color:var(--live1); }} .badge.y {{ border-color:rgba(240,167,66,.3); color:var(--live2); }} .badge.b {{ border-color:rgba(45,95,168,.3); color:var(--nonlive); }}
  .mini-stats {{ display:flex; gap:8px; margin-top:14px; }}
  .mini-stat {{ flex:1; background:var(--surface2); border-radius:6px; padding:10px 12px; text-align:center; }}
  .mini-stat-lbl {{ font-size:10px; color:var(--text-dim); margin-bottom:3px; }} .mini-stat-val {{ font-size:18px; font-weight:500; color:var(--text); }} .mini-stat-sub {{ font-size:10px; color:var(--text-dim); margin-top:2px; }}
  .chart-wrap {{ position:relative; width:100%; }}
  .legend {{ display:flex; gap:16px; margin-bottom:12px; flex-wrap:wrap; }}
  .legend-item {{ display:flex; align-items:center; gap:6px; font-size:11px; color:var(--text-muted); }}
  .legend-dot {{ width:10px; height:10px; border-radius:2px; flex-shrink:0; }}
  .reason-row,.break-row {{ display:flex; align-items:center; gap:10px; padding:7px 0; border-bottom:1px solid var(--border); font-size:12px; }}
  .reason-row:last-child,.break-row:last-child {{ border-bottom:none; }}
  .reason-name,.break-name {{ flex:1; color:var(--text-muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .reason-bar-wrap,.break-bar-wrap {{ width:86px; height:4px; background:var(--surface2); border-radius:2px; overflow:hidden; }}
  .reason-bar,.break-bar {{ height:100%; border-radius:2px; }}
  .reason-cnt,.break-num {{ font-family:var(--mono); font-size:11px; color:var(--text); min-width:38px; text-align:right; }}
  .reason-pct,.break-pct {{ font-family:var(--mono); font-size:10px; color:var(--text-dim); min-width:42px; text-align:right; }}
  .break-orders {{ font-family:var(--mono); font-size:10px; color:var(--text-dim); min-width:72px; text-align:right; }}
  .insights {{ display:flex; flex-direction:column; gap:10px; margin-top:0; }}
  .insight {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px 20px; font-size:13px; color:var(--text-muted); line-height:1.7; display:flex; gap:14px; align-items:flex-start; }}
  .insight-icon {{ font-family:var(--mono); font-size:10px; color:var(--accent); letter-spacing:.05em; flex-shrink:0; padding-top:3px; }}
  .insight strong {{ color:var(--text); font-weight:500; }}
  footer {{ border-top:1px solid var(--border); margin:0 60px; padding:24px 0; font-family:var(--mono); font-size:10px; color:var(--text-dim); display:flex; justify-content:space-between; gap:16px; }}
  .empty-note {{ font-size:12px; color:var(--text-dim); padding:12px 0; }}
  @keyframes fadeUp {{ from {{ opacity:0; transform:translateY(16px); }} to {{ opacity:1; transform:translateY(0); }} }}
</style>
</head>
<body>
<header>
  <div class="header-tag">Cancel Order Analytics · {html.escape(ctx['range_label'])}</div>
  <h1>取消订单<br>分析报告</h1>
  <p class="header-sub">订单总表清洗 · Order ID 去重 · Created Time 直播归因 · SKU/甲型与产品链接拆解</p>
  <div class="header-meta">
    <strong>{fmt_num(ctx['cancel_orders'])}</strong>
    Cancelled Orders<br>
    占总订单 {ctx['cancel_rate']:.1f}%
  </div>
</header>
<main>
  <div class="section">
    <div class="section-label"><span>01</span>总览 Overview</div>
    <div class="stat-grid">
      <div class="stat green"><div class="stat-lbl">总订单 Total Orders</div><div class="stat-val green">{fmt_num(ctx['total_orders'])}</div><div class="stat-sub">按 Order ID 去重</div></div>
      <div class="stat red"><div class="stat-lbl">Cancelled Orders</div><div class="stat-val red">{fmt_num(ctx['cancel_orders'])}</div><div class="stat-sub">B column Order Status = Cancelled/Canceled</div></div>
      <div class="stat amber"><div class="stat-lbl">Cancel Rate</div><div class="stat-val amber">{ctx['cancel_rate']:.1f}%</div><div class="stat-sub">Cancelled / Total Orders</div></div>
      <div class="stat blue"><div class="stat-lbl">Cancelled SKU Units</div><div class="stat-val blue">{fmt_num(ctx['cancel_sku_metric'])}</div><div class="stat-sub">{html.escape(ctx['metric_label'])}</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-label"><span>02</span>直播购买归因 · Based on Created Time</div>
    <div class="stat-grid">
      <div class="stat red"><div class="stat-lbl">直播① Cancelled</div><div class="stat-val red">{fmt_num(ctx['live1_cancel'])}</div><div class="stat-sub">{ctx['live1_start']}:00–{ctx['live1_end']}:00 · 占 cancelled {fmt_pct(ctx['live1_cancel'], ctx['cancel_orders'])}</div></div>
      <div class="stat amber"><div class="stat-lbl">直播② Cancelled</div><div class="stat-val amber">{fmt_num(ctx['live2_cancel'])}</div><div class="stat-sub">{ctx['live2_start']}:00–{ctx['live2_end']}:00 · 占 cancelled {fmt_pct(ctx['live2_cancel'], ctx['cancel_orders'])}</div></div>
      <div class="stat blue"><div class="stat-lbl">非直播 Cancelled</div><div class="stat-val blue">{fmt_num(ctx['nonlive_cancel'])}</div><div class="stat-sub">Created Time 不在直播时段</div></div>
      <div class="stat purple"><div class="stat-lbl">直播创建订单 Cancel Rate</div><div class="stat-val purple">{ctx['live_cancel_rate']:.1f}%</div><div class="stat-sub">非直播 {ctx['nonlive_cancel_rate']:.1f}%</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-label"><span>03</span>Created Time vs Cancelled Time · 每小时分布</div>
    <div class="two-col">
      <div class="panel">
        <div class="panel-title">Cancelled Orders 的创建时间 <span class="badge r">用于直播归因</span></div>
        <div class="legend"><div class="legend-item"><div class="legend-dot" style="background:var(--live1)"></div>直播①</div><div class="legend-item"><div class="legend-dot" style="background:var(--live2)"></div>直播②</div><div class="legend-item"><div class="legend-dot" style="background:var(--nonlive)"></div>非直播</div></div>
        <div class="chart-wrap" style="height:210px"><canvas id="createdHour"></canvas></div>
        <div class="mini-stats"><div class="mini-stat"><div class="mini-stat-lbl">创建峰值</div><div class="mini-stat-val" style="color:var(--live1)">{ctx['created_peak_label']}</div><div class="mini-stat-sub">{fmt_num(ctx['created_peak_val'])}单</div></div><div class="mini-stat"><div class="mini-stat-lbl">直播归因占比</div><div class="mini-stat-val">{fmt_pct(ctx['live_cancel'], ctx['cancel_orders'])}</div><div class="mini-stat-sub">按 Created Time</div></div></div>
      </div>
      <div class="panel">
        <div class="panel-title">实际取消时间 <span class="badge b">仅看 Cancelled Time 峰值</span></div>
        <div class="chart-wrap" style="height:210px"><canvas id="cancelledHour"></canvas></div>
        <div class="mini-stats"><div class="mini-stat"><div class="mini-stat-lbl">取消峰值</div><div class="mini-stat-val" style="color:var(--nonlive)">{ctx['cancelled_peak_label']}</div><div class="mini-stat-sub">{fmt_num(ctx['cancelled_peak_val'])}单</div></div><div class="mini-stat"><div class="mini-stat-lbl">缺失 Cancelled Time</div><div class="mini-stat-val">{fmt_num(ctx['missing_cancelled_time'])}</div><div class="mini-stat-sub">不进入右图</div></div></div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-label"><span>04</span>取消原因 Cancel Reason</div>
    <div class="full-panel">
      {make_reason_rows(ctx['reason_df'], 'var(--live1)')}
    </div>
  </div>

  <div class="section">
    <div class="section-label"><span>05</span>Cancelled Orders · 甲型 / SKU Breakdown</div>
    <div class="full-panel">
      <div class="panel-title">各个甲型个数和占比 <span class="badge r">{html.escape(ctx['metric_label'])}</span></div>
      {make_breakdown_rows(ctx['style_breakdown'], 'Nail Style', ctx['metric_col'], 'var(--live1)')}
    </div>
  </div>

  <div class="section">
    <div class="section-label"><span>06</span>Cancelled Orders · H Column 产品链接 Breakdown</div>
    <div class="full-panel">
      <div class="panel-title">各个产品链接 / Product Name 个数和占比 <span class="badge b">H column</span></div>
      {make_breakdown_rows(ctx['product_breakdown'], 'Product Link / Product Name', ctx['metric_col'], 'var(--nonlive)')}
    </div>
  </div>

  <div class="section">
    <div class="section-label"><span>07</span>关键洞察 Key Insights</div>
    <div class="insights">{insights_html}</div>
  </div>
</main>
<footer>
  <span>数据来源：{html.escape(ctx['source_name'])} · 日期口径：Created Time · 订单口径：Order ID unique</span>
  <span>直播归因：Created Time in {ctx['live1_start']}:00–{ctx['live1_end']}:00 / {ctx['live2_start']}:00–{ctx['live2_end']}:00</span>
</footer>
<script>
const C = Chart;
const hrs = Array.from({{length:24}},(_,i)=>i);
function barColor(h){{
  if(h>={ctx['live1_start']} && h<{ctx['live1_end']}) return '#d44a1e';
  if(h>={ctx['live2_start']} && h<{ctx['live2_end']}) return '#c8840a';
  return '#2d5fa8';
}}
const colors = hrs.map(barColor);
function commonOpts(maxVal, step){{
  return {{ responsive:true, maintainAspectRatio:false, plugins:{{ legend:{{display:false}} }},
    scales:{{
      x:{{ ticks:{{autoSkip:false,maxRotation:0,font:{{size:9}},color:'#9a9ca3'}}, grid:{{color:'rgba(0,0,0,0.05)'}}, border:{{color:'rgba(0,0,0,0.08)'}} }},
      y:{{ beginAtZero:true, max:maxVal, ticks:{{stepSize:step,font:{{size:10}},color:'#9a9ca3'}}, grid:{{color:'rgba(0,0,0,0.05)'}}, border:{{color:'rgba(0,0,0,0.08)'}} }}
    }}
  }};
}}
new C(document.getElementById('createdHour'), {{
  type:'bar',
  data:{{ labels:hrs.map(h=>h+'h'), datasets:[{{ data:{json.dumps(ctx['created_hour_counts'])}, backgroundColor:colors, borderRadius:3 }}] }},
  options:commonOpts({created_max}, {5 if created_max >= 10 else 1})
}});
new C(document.getElementById('cancelledHour'), {{
  type:'bar',
  data:{{ labels:hrs.map(h=>h+'h'), datasets:[{{ data:{json.dumps(ctx['cancelled_hour_counts'])}, backgroundColor:'#2d5fa8', borderRadius:3 }}] }},
  options:commonOpts({cancelled_max}, {5 if cancelled_max >= 10 else 1})
}});
</script>
</body>
</html>'''
    return html_doc


def make_excel_download(all_orders, cancel_orders, cancel_lines, style_breakdown, product_breakdown, reason_df, ctx) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary = pd.DataFrame([
            ["Date Range", ctx["range_label"]],
            ["Total Orders", ctx["total_orders"]],
            ["Cancelled Orders", ctx["cancel_orders"]],
            ["Cancel Rate", f"{ctx['cancel_rate']:.1f}%"],
            ["Cancelled SKU Units", ctx["cancel_sku_metric"]],
            ["Metric", ctx["metric_label"]],
            ["Live Cancelled by Created Time", ctx["live_cancel"]],
            ["Non-live Cancelled by Created Time", ctx["nonlive_cancel"]],
            ["Live Created Order Cancel Rate", f"{ctx['live_cancel_rate']:.1f}%"],
            ["Non-live Created Order Cancel Rate", f"{ctx['nonlive_cancel_rate']:.1f}%"],
        ], columns=["Metric", "Value"])
        summary.to_excel(writer, index=False, sheet_name="Summary")
        all_orders.to_excel(writer, index=False, sheet_name="Order Level All")
        cancel_orders.to_excel(writer, index=False, sheet_name="Cancelled Orders")
        cancel_lines.to_excel(writer, index=False, sheet_name="Cancelled SKU Lines")
        style_breakdown.to_excel(writer, index=False, sheet_name="Nail Style Breakdown")
        product_breakdown.to_excel(writer, index=False, sheet_name="Product Link Breakdown")
        reason_df.to_excel(writer, index=False, sheet_name="Cancel Reasons")
        pd.DataFrame({
            "Hour": list(range(24)),
            "Cancelled Orders Created Time": ctx["created_hour_counts"],
            "Cancelled Orders Cancelled Time": ctx["cancelled_hour_counts"],
        }).to_excel(writer, index=False, sheet_name="Hourly")

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#F3F4F6", "border": 1})
        for sheet in writer.sheets.values():
            sheet.freeze_panes(1, 0)
            sheet.set_row(0, None, header_fmt)
            sheet.set_column(0, 0, 26)
            sheet.set_column(1, 20, 18)
    return output.getvalue()


# =========================
# Streamlit UI
# =========================
st.title("💅 Cancelled Orders Report Generator")
st.caption("上传 TikTok Shop 订单总表，程序会自动清洗出 B column `Order Status = Cancelled/Canceled` 的订单，并按 Order ID 去重生成报告。")

with st.expander("这版程序的核心口径", expanded=True):
    st.markdown(
        """
- **上传文件**：订单总表，不再上传 cancelled order 表。
- **Cancelled 识别**：B column `Order Status` 等于 `Cancelled` 或 `Canceled`。
- **订单口径**：一个 `Order ID` 只算一个订单；一个 `Order ID` 只算一个 cancelled order。
- **直播归因**：使用 **AB column `Created Time`** 判断是否属于直播间购买，而不是 `Cancelled Time`。
- **取消峰值**：使用 `Cancelled Time` 单独分析“顾客什么时候取消”，但它不用于直播归因。
- **甲型 / 产品链接拆解**：基于 cancelled orders 对应的 SKU 行做统计，因为一个订单可能有多个 SKU。
        """
    )

uploaded_file = st.file_uploader("上传订单总表 CSV / Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("请先上传 TikTok Shop 导出的订单总表。")
    st.stop()

try:
    raw = read_uploaded_file(uploaded_file)
except Exception as e:
    st.error(f"文件读取失败：{e}")
    st.stop()

raw.columns = [clean_column_name(c) for c in raw.columns]
missing = validate_columns(raw)
if missing:
    st.error("缺少必要字段：" + ", ".join(missing))
    st.write("当前识别到的字段：", list(raw.columns))
    st.stop()

with st.sidebar:
    st.header("Report Settings")
    st.caption("日期筛选默认使用 Created Time，因为总订单和直播归因都应该按下单创建时间判断。")

    metric_mode_label = st.radio(
        "甲型 / 产品链接统计口径",
        options=["按 Quantity 汇总（推荐）", "按 SKU 行数汇总"],
        index=0,
        help="如果同一 SKU 行 Quantity > 1，按 Quantity 汇总会更接近实际件数；按 SKU 行数则一行算 1。",
    )
    metric_mode = "quantity" if metric_mode_label.startswith("按 Quantity") else "rows"

    live1_col, live2_col = st.columns(2)
    with live1_col:
        live1_start = st.number_input("直播①开始", min_value=0, max_value=23, value=10, step=1)
        live1_end = st.number_input("直播①结束", min_value=1, max_value=24, value=18, step=1)
    with live2_col:
        live2_start = st.number_input("直播②开始", min_value=0, max_value=23, value=19, step=1)
        live2_end = st.number_input("直播②结束", min_value=1, max_value=24, value=23, step=1)

    top_n = st.slider("Breakdown 显示 Top N", min_value=5, max_value=30, value=10, step=1)

if not (live1_start < live1_end and live2_start < live2_end):
    st.error("直播开始时间必须小于结束时间。")
    st.stop()

lines = prepare_line_level(raw, metric_mode=metric_mode)
missing_created = int(lines["Created Datetime"].isna().sum())
valid_created = lines[lines["Created Datetime"].notna()].copy()
if valid_created.empty:
    st.error("Created Time 无法解析，无法继续。请检查 AB column Created Time 格式。")
    st.stop()

min_date = valid_created["Created Datetime"].dt.date.min()
max_date = valid_created["Created Datetime"].dt.date.max()

with st.sidebar:
    date_range = st.date_input(
        "选择 Created Time 日期区间",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

selected_lines = filter_by_created_date(lines, start_date, end_date)
if selected_lines.empty:
    st.warning("当前日期区间内没有订单。")
    st.stop()

all_orders = build_order_level(selected_lines, live1_start, live1_end, live2_start, live2_end)
cancel_orders = all_orders[all_orders["Is Cancelled"]].copy()
cancel_order_ids = set(cancel_orders["Order ID"])
cancel_lines = selected_lines[selected_lines["Order ID"].isin(cancel_order_ids)].copy()

metric_col = "Units" if metric_mode == "quantity" else "SKU Rows"
metric_label = "按 Quantity 汇总" if metric_mode == "quantity" else "按 SKU 行数汇总"
style_breakdown = make_breakdown(cancel_lines, "Nail Style", top_n=top_n, metric_label=metric_col)
product_breakdown = make_breakdown(cancel_lines, "Product Link / Product Name", top_n=top_n, metric_label=metric_col)
reason_df = value_count_table(cancel_orders, "Cancel Reason Clean", len(cancel_orders), top_n=top_n)

created_hour_counts = hourly_counts_from_series(cancel_orders["Created Datetime"] if not cancel_orders.empty else pd.Series(dtype="datetime64[ns]"))
cancelled_hour_counts = hourly_counts_from_series(cancel_orders["Cancelled Datetime"] if not cancel_orders.empty else pd.Series(dtype="datetime64[ns]"))
created_peak_label, created_peak_val = peak_label(created_hour_counts)
cancelled_peak_label, cancelled_peak_val = peak_label(cancelled_hour_counts)

live1_cancel = int((cancel_orders["Live Segment by Created Time"] == "直播①").sum()) if not cancel_orders.empty else 0
live2_cancel = int((cancel_orders["Live Segment by Created Time"] == "直播②").sum()) if not cancel_orders.empty else 0
live_cancel = live1_cancel + live2_cancel
nonlive_cancel = len(cancel_orders) - live_cancel

live1_all = int((all_orders["Live Segment by Created Time"] == "直播①").sum()) if not all_orders.empty else 0
live2_all = int((all_orders["Live Segment by Created Time"] == "直播②").sum()) if not all_orders.empty else 0
live_all = live1_all + live2_all
nonlive_all = len(all_orders) - live_all
live1_cancel_rate = pct(live1_cancel, live1_all)
live2_cancel_rate = pct(live2_cancel, live2_all)
live_cancel_rate = pct(live_cancel, live_all)
nonlive_cancel_rate = pct(nonlive_cancel, nonlive_all)

cancel_sku_metric = float(cancel_lines["Metric Units"].sum()) if not cancel_lines.empty else 0
missing_cancelled_time = int(cancel_orders["Cancelled Datetime"].isna().sum()) if not cancel_orders.empty else 0

range_label = f"{start_date.strftime('%Y/%m/%d')}–{end_date.strftime('%Y/%m/%d')}"
ctx = {
    "source_name": uploaded_file.name,
    "report_title": f"{range_label} Cancelled Orders Report",
    "range_label": range_label,
    "total_orders": len(all_orders),
    "cancel_orders": len(cancel_orders),
    "cancel_rate": pct(len(cancel_orders), len(all_orders)),
    "cancel_sku_metric": cancel_sku_metric,
    "metric_label": metric_label,
    "metric_col": metric_col,
    "live1_start": int(live1_start),
    "live1_end": int(live1_end),
    "live2_start": int(live2_start),
    "live2_end": int(live2_end),
    "live1_cancel": live1_cancel,
    "live2_cancel": live2_cancel,
    "live_cancel": live_cancel,
    "nonlive_cancel": nonlive_cancel,
    "live1_all": live1_all,
    "live2_all": live2_all,
    "live_all": live_all,
    "nonlive_all": nonlive_all,
    "live1_cancel_rate": live1_cancel_rate,
    "live2_cancel_rate": live2_cancel_rate,
    "live_cancel_rate": live_cancel_rate,
    "nonlive_cancel_rate": nonlive_cancel_rate,
    "created_hour_counts": created_hour_counts,
    "cancelled_hour_counts": cancelled_hour_counts,
    "created_peak_label": created_peak_label,
    "created_peak_val": created_peak_val,
    "cancelled_peak_label": cancelled_peak_label,
    "cancelled_peak_val": cancelled_peak_val,
    "missing_cancelled_time": missing_cancelled_time,
    "reason_df": reason_df,
    "style_breakdown": style_breakdown,
    "product_breakdown": product_breakdown,
}
ctx["insights"] = build_insights(
    all_orders, cancel_orders, cancel_lines, style_breakdown, product_breakdown,
    start_date, end_date, live1_start, live1_end, live2_start, live2_end,
)

html_report = build_html_report(ctx)
excel_bytes = make_excel_download(all_orders, cancel_orders, cancel_lines, style_breakdown, product_breakdown, reason_df, ctx)

# =========================
# Display in Streamlit
# =========================
if missing_created:
    st.warning(f"有 {missing_created:,} 行 Created Time 无法解析，已从日期筛选和报告中排除。")

st.subheader("核心结果")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Orders", fmt_num(len(all_orders)))
k2.metric("Cancelled Orders", fmt_num(len(cancel_orders)))
k3.metric("Cancel Rate", f"{ctx['cancel_rate']:.1f}%")
k4.metric("Live-attributed Cancelled", fmt_num(live_cancel), help="按 Created Time 判断是否在直播时段创建")
k5.metric("Cancelled SKU Units", fmt_num(cancel_sku_metric), help=metric_label)

st.subheader("直播归因判断：按 Created Time")
live_summary = pd.DataFrame([
    ["直播①", live1_cancel, live1_all, live1_cancel_rate, fmt_pct(live1_cancel, len(cancel_orders))],
    ["直播②", live2_cancel, live2_all, live2_cancel_rate, fmt_pct(live2_cancel, len(cancel_orders))],
    ["直播合计", live_cancel, live_all, live_cancel_rate, fmt_pct(live_cancel, len(cancel_orders))],
    ["非直播", nonlive_cancel, nonlive_all, nonlive_cancel_rate, fmt_pct(nonlive_cancel, len(cancel_orders))],
], columns=["Segment", "Cancelled Orders", "Total Created Orders in Segment", "Segment Cancel Rate", "% of Cancelled Orders"])
st.dataframe(live_summary, use_container_width=True, hide_index=True)

st.subheader("Breakdowns")
tab1, tab2, tab3, tab4 = st.tabs(["Cancel Reasons", "甲型 / SKU", "H Column 产品链接", "订单级 Cancelled 明细"])
with tab1:
    st.dataframe(reason_df, use_container_width=True, hide_index=True)
with tab2:
    st.dataframe(style_breakdown, use_container_width=True, hide_index=True)
with tab3:
    st.dataframe(product_breakdown, use_container_width=True, hide_index=True)
with tab4:
    st.dataframe(cancel_orders, use_container_width=True, hide_index=True)

st.subheader("HTML Report Preview")
components.html(html_report, height=1100, scrolling=True)

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "下载 HTML Report",
        data=html_report.encode("utf-8"),
        file_name=f"cancelled_orders_report_{start_date}_{end_date}.html",
        mime="text/html",
        use_container_width=True,
    )
with d2:
    st.download_button(
        "下载清洗后 Excel",
        data=excel_bytes,
        file_name=f"cancelled_orders_cleaned_{start_date}_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
