import html
import json
import re
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
    page_title="Cancelled + Returned Orders Report Generator",
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

RETURN_TARGET_PRODUCT_LINKS = [
    "Dreamwear",
    "Top Trend",
    "Next Gen",
    "New Drop",
    "Final Sale",
    "Square",
    "Stiletto",
    "Almond",
    "Spring Bloom",
    "Summer Shine",
    "Best Seller",
    "Organizer Binder",
    "TOOLKITS",
    "BUY 4 GET 1 FREE",
]


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


def _join_unique_by_order_fast(df: pd.DataFrame, col: str, max_items: int = 50) -> pd.Series:
    """Fast replacement for per-group join_unique. Returns Series indexed by Order ID."""
    if df.empty or col not in df.columns:
        return pd.Series(dtype=object)
    x = df[["Order ID", col]].copy()
    x[col] = (
        x[col]
        .fillna("")
        .astype(str)
        .str.replace("\t", "", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )
    x = x[(x[col] != "") & (~x[col].str.lower().isin(["nan", "nat", "none"]))]
    if x.empty:
        return pd.Series(dtype=object)
    x = x.drop_duplicates(["Order ID", col])
    x["__rank"] = x.groupby("Order ID", sort=False).cumcount()
    x = x[x["__rank"] < max_items].drop(columns="__rank")
    return x.groupby("Order ID", sort=False)[col].agg(lambda vals: "; ".join(vals))


def _classify_live_segment_series(hour_series: pd.Series, live1_start, live1_end, live2_start, live2_end) -> pd.Series:
    h = pd.to_numeric(hour_series, errors="coerce")
    out = pd.Series("非直播", index=hour_series.index, dtype=object)
    out[(h >= live1_start) & (h < live1_end)] = "直播①"
    out[(h >= live2_start) & (h < live2_end)] = "直播②"
    out[h.isna()] = "Unknown"
    return out


def build_order_level(lines: pd.DataFrame, live1_start, live1_end, live2_start, live2_end) -> pd.DataFrame:
    """
    Build one-row-per-Order-ID table.
    Optimized because the all-order file can be large and Streamlit reruns the app on every interaction.
    """
    if lines.empty:
        return pd.DataFrame()

    tmp = lines.copy()
    tmp["__status_cancelled_int"] = tmp["Is Cancelled Line"].astype("int8")

    # Use vectorized/built-in aggregations. For TikTok exports, order-level fields are
    # duplicated across SKU rows, so first/min/sum are sufficient and far faster than
    # Python custom functions per Order ID.
    agg = {
        "Created Datetime": "min",
        "Cancelled Datetime": "first",
        "Order Status Clean": "first",
        "__status_cancelled_int": "max",
        "Cancel Reason Clean": "first",
        "Metric Units": "sum",
        "SKU Row Count": "sum",
        "Quantity Parsed": "sum",
    }
    if "Order Amount Parsed" in tmp.columns:
        agg["Order Amount Parsed"] = "first"
    if "SKU Subtotal After Discount Parsed" in tmp.columns:
        agg["SKU Subtotal After Discount Parsed"] = "sum"

    od = tmp.groupby("Order ID", as_index=False, sort=False).agg(agg)

    # Detail columns are only for display/export. Build them with a faster de-duped join.
    for col in ["Nail Style", "Product Link / Product Name", "Seller SKU", "Variation"]:
        if col in tmp.columns:
            joined = _join_unique_by_order_fast(tmp, col)
            od = od.merge(joined.rename(col), on="Order ID", how="left")
            if col in ["Nail Style", "Product Link / Product Name"]:
                od[col] = od[col].fillna("Unknown")
            else:
                od[col] = od[col].fillna("")

    od["Is Cancelled"] = od["__status_cancelled_int"].eq(1)
    od["Order Status Final"] = np.where(od["Is Cancelled"], "Cancelled", od["Order Status Clean"])
    od["Created Hour"] = od["Created Datetime"].dt.hour
    od["Created Weekday Num"] = od["Created Datetime"].dt.weekday
    od["Created Day Type"] = np.where(od["Created Weekday Num"] >= 5, "周末 Weekend", "工作日 Weekday")
    od.loc[od["Created Datetime"].isna(), "Created Day Type"] = "Unknown"
    od["Live Segment by Created Time"] = _classify_live_segment_series(
        od["Created Hour"], live1_start, live1_end, live2_start, live2_end
    )
    od["Is Live by Created Time"] = od["Live Segment by Created Time"].isin(["直播①", "直播②"])
    od["Cancelled Hour"] = od["Cancelled Datetime"].dt.hour
    od["Cancelled Day Type"] = np.where(
        od["Cancelled Datetime"].dt.weekday >= 5, "周末 Weekend", "工作日 Weekday"
    )
    od.loc[od["Cancelled Datetime"].isna(), "Cancelled Day Type"] = "Unknown"
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


def display_table(df: pd.DataFrame, drop_sku_row_count: bool = False) -> pd.DataFrame:
    """Format tables for Streamlit/Excel display without changing internal calculation columns."""
    out = df.copy()
    # For Nail Style / Product Link breakdowns, remove the duplicate SKU Row Count column.
    # Keep Order Count because it shows how many unique cancelled orders are involved.
    if drop_sku_row_count and "SKU Row Count" in out.columns:
        out = out.drop(columns=["SKU Row Count"])
    out = out.rename(columns={"Pct": "占比 %"})
    return out


def strip_seller_sku_size_suffix(x) -> str:
    """H column Seller SKU: remove final size suffix like -S / -M / -L / -XS / -XL."""
    s = normalize_text(x, default="Unknown")
    if s == "Unknown":
        return s
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(?i)([-_\s]+)(XS|S|M|L|XL|XXL|XXXL)$", "", s)
    return s or "Unknown"


def strip_sku_name_size_suffix(x) -> str:
    """J column SKU Name: style name, remove final size token such as ', S'."""
    s = normalize_text(x, default="Unknown")
    if s == "Unknown":
        return s
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) >= 2 and parts[-1].upper() in SIZE_TOKENS:
        s = ", ".join(parts[:-1]).strip()
    s = re.sub(r"(?i)\s*[-_/]\s*(XS|S|M|L|XL|XXL|XXXL)$", "", s).strip()
    return s or "Unknown"


def is_no_longer_needed(reason) -> bool:
    return normalize_text(reason, default="").strip().lower() == "no longer needed"


def is_request_cancelled_status(x) -> bool:
    s = normalize_text(x, default="").strip().lower()
    return "request" in s and ("cancelled" in s or "canceled" in s or "cancel" in s)


def is_refund_only_type(x) -> bool:
    s = normalize_text(x, default="").strip().lower().replace("-", " ")
    return s == "refund only"


def display_return_breakdown_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={
        "Pct": "占比 %",
        "Return Style English (Catalog)": "款式英文名",
        "Return Product Link (I Product Name)": "退货产品链接",
        "Product Link": "指定产品链接",
        "Returned Units": "退货件数",
        "Return Package Count": "退货包裹数",
        "SKU Rows": "SKU行数",
    })
    return out


def build_catalog_sku_name_map(catalog_raw: pd.DataFrame) -> dict:
    """产品图册：SKU -> 款式英文名称。SKU 会去掉尺码后缀后匹配。"""
    if catalog_raw is None or catalog_raw.empty:
        return {}
    df = catalog_raw.copy()
    df.columns = [clean_column_name(c) for c in df.columns]
    if "SKU" not in df.columns or "款式英文名称" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        sku = strip_seller_sku_size_suffix(row.get("SKU", ""))
        name = normalize_text(row.get("款式英文名称", ""), default="")
        if sku and sku != "Unknown" and name:
            out[sku] = name
    return out


def apply_catalog_style_names(return_lines: pd.DataFrame, catalog_map: dict) -> pd.DataFrame:
    """Use H-column Seller SKU base code to map to catalog English style name; fallback to J-column SKU Name."""
    df = return_lines.copy()
    if df.empty:
        df["Return Style English (Catalog)"] = []
        return df
    if not catalog_map:
        df["Return Style English (Catalog)"] = df.get("Return Style (J SKU Name)", pd.Series(["Unknown"] * len(df), index=df.index))
        return df
    df["Return Style English (Catalog)"] = df["Return SKU (H no size)"].map(catalog_map)
    fallback = df.get("Return Style (J SKU Name)", pd.Series(["Unknown"] * len(df), index=df.index))
    df["Return Style English (Catalog)"] = df["Return Style English (Catalog)"].fillna(fallback)
    df["Return Style English (Catalog)"] = df["Return Style English (Catalog)"].apply(lambda x: normalize_text(x, default="Unknown"))
    return df


def make_return_specific_product_link_share(lines: pd.DataFrame, target_links=None) -> pd.DataFrame:
    """For specified product links/collections, calculate return share against total returned units."""
    if target_links is None:
        target_links = RETURN_TARGET_PRODUCT_LINKS
    if lines.empty or "Return Product Link (I Product Name)" not in lines.columns:
        return pd.DataFrame(columns=["Product Link", "Returned Units", "Return Package Count", "SKU Rows", "Pct"])
    total_units = lines["Return Quantity Parsed"].sum()
    product_lower = lines["Return Product Link (I Product Name)"].fillna("").astype(str).str.lower()
    rows = []
    for label in target_links:
        label_lower = str(label).lower()
        mask = product_lower.str.contains(re.escape(label_lower), na=False, regex=True)
        sub = lines[mask]
        units = float(sub["Return Quantity Parsed"].sum()) if not sub.empty else 0
        pkg_count = int(sub["Return Package Key"].nunique()) if not sub.empty else 0
        sku_rows = int(sub["Returned SKU Row Count"].sum()) if not sub.empty else 0
        rows.append([label, units, pkg_count, sku_rows, pct(units, total_units)])
    return pd.DataFrame(rows, columns=["Product Link", "Returned Units", "Return Package Count", "SKU Rows", "Pct"])


# =========================
# Returned order / package helpers
# =========================
def choose_return_package_key(row) -> str:
    """One returned package = tracking number when available; fallback to return order/order id."""
    tracking = normalize_text(row.get("Return Logistics Tracking ID", ""), default="")
    return_order_id = normalize_text(row.get("Return Order ID", ""), default="")
    order_id = normalize_text(row.get("Order ID", ""), default="")
    if tracking:
        return f"TRACKING：{tracking}"
    if return_order_id:
        return f"RETURN_ORDER：{return_order_id}"
    if order_id:
        return f"ORDER：{order_id}"
    return "UNKNOWN_PACKAGE"


def prepare_return_line_level(return_raw: pd.DataFrame, all_order_lookup: pd.DataFrame,
                              live1_start, live1_end, live2_start, live2_end) -> pd.DataFrame:
    """
    Returned Order file is all considered returned. It is SKU-line level.
    Created Time is pulled by matching Order ID back to the uploaded all-order table.
    """
    df = return_raw.copy()
    df.columns = [clean_column_name(c) for c in df.columns]
    if "Order ID" not in df.columns:
        raise ValueError("Returned Order 表缺少必要字段：Order ID")

    df["Order ID"] = df["Order ID"].apply(stringify_id)
    df = df[df["Order ID"] != ""].copy()
    if df.empty:
        return df

    needed_cols = [
        "Return Order ID", "Return Logistics Tracking ID", "Return Reason", "Return Type",
        "Return Status", "Return Sub Status", "Product Name", "SKU Name", "Seller SKU",
        "Buyer Username", "Buyer Note"
    ]
    for c in needed_cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].apply(lambda x: normalize_text(x, default=""))

    if "Return Quantity" in df.columns:
        rq = parse_number_series(df["Return Quantity"], default=np.nan).fillna(1)
        rq = rq.where(rq > 0, 1)
    else:
        rq = pd.Series([1] * len(df), index=df.index)
    df["Return Quantity Parsed"] = rq
    df["Returned SKU Row Count"] = 1
    df["Return Package Key"] = df.apply(choose_return_package_key, axis=1)

    # H column = Seller SKU, ignore size suffix; J column = SKU Name;
    # N = Return Reason; S = Return Sub Status; Q = Return Logistics Tracking ID; L = Return Type.
    df["Return SKU (H no size)"] = df["Seller SKU"].apply(strip_seller_sku_size_suffix)
    df["Return Style (J SKU Name)"] = df["SKU Name"].apply(strip_sku_name_size_suffix)
    df["Return Style English (Catalog)"] = df["Return Style (J SKU Name)"]
    df["Return Product Link (I Product Name)"] = df["Product Name"].apply(lambda x: normalize_text(x, default="Unknown"))
    df["Return Reason Clean"] = df["Return Reason"].apply(lambda x: normalize_text(x, default="Unknown"))
    df["Return Type Clean"] = df["Return Type"].apply(lambda x: normalize_text(x, default="Unknown"))
    df["Return Sub Status Clean"] = df["Return Sub Status"].apply(lambda x: normalize_text(x, default="Unknown"))
    df["Seller Fault Flag"] = ~df["Return Reason Clean"].apply(is_no_longer_needed)
    df["Return Reason Attribution"] = np.where(df["Seller Fault Flag"], "Seller Fault", "No Longer Needed")
    df["Request Cancelled Flag"] = df["Return Sub Status Clean"].apply(is_request_cancelled_status)
    df["Has Return Tracking Flag"] = df["Return Logistics Tracking ID"].apply(lambda x: bool(normalize_text(x, default="")))
    df["Refund Only Flag"] = df["Return Type Clean"].apply(is_refund_only_type)

    if "Created Time" in df.columns:
        df["Created Datetime From Return File"] = parse_datetime_series(df["Created Time"])
    else:
        df["Created Datetime From Return File"] = pd.NaT

    lookup = all_order_lookup[["Order ID", "Created Datetime"]].drop_duplicates("Order ID").copy()
    df = df.merge(lookup, on="Order ID", how="left")
    df["Created Datetime"] = df["Created Datetime"].fillna(df["Created Datetime From Return File"])
    df["Created Hour"] = df["Created Datetime"].dt.hour
    df["Live Segment by Created Time"] = df["Created Hour"].apply(
        lambda h: classify_live_segment(h, live1_start, live1_end, live2_start, live2_end)
    )
    df.loc[df["Created Datetime"].isna(), "Live Segment by Created Time"] = "Unknown"
    df["Is Live by Created Time"] = df["Live Segment by Created Time"].isin(["直播①", "直播②"])
    return df

def build_return_package_level(return_lines: pd.DataFrame) -> pd.DataFrame:
    if return_lines.empty:
        return pd.DataFrame()
    tmp = return_lines.copy()
    tmp["Seller Fault Int"] = tmp["Seller Fault Flag"].astype(int)
    tmp["Request Cancelled Int"] = tmp["Request Cancelled Flag"].astype(int)
    tmp["Has Return Tracking Int"] = tmp["Has Return Tracking Flag"].astype(int)
    tmp["Refund Only Int"] = tmp["Refund Only Flag"].astype(int)

    pkg = (
        tmp.groupby("Return Package Key", as_index=False)
        .agg(
            **{
                "Order IDs": ("Order ID", join_unique),
                "Return Order IDs": ("Return Order ID", join_unique),
                "Return Logistics Tracking ID": ("Return Logistics Tracking ID", mode_or_first),
                "Created Datetime": ("Created Datetime", first_non_null),
                "Return Reason": ("Return Reason Clean", mode_or_first),
                "Return Type": ("Return Type Clean", mode_or_first),
                "Return Status": ("Return Status", mode_or_first),
                "Return Sub Status": ("Return Sub Status Clean", mode_or_first),
                "Returned SKU Rows": ("Returned SKU Row Count", "sum"),
                "Return Quantity": ("Return Quantity Parsed", "sum"),
                "Product Names": ("Product Name", join_unique),
                "Return Product Links I": ("Return Product Link (I Product Name)", join_unique),
                "Return SKUs H no size": ("Return SKU (H no size)", join_unique),
                "Return Style English Catalog": ("Return Style English (Catalog)", join_unique),
                "Return Styles J SKU Name": ("Return Style (J SKU Name)", join_unique),
                "Seller SKUs Raw": ("Seller SKU", join_unique),
                "SKU Names Raw": ("SKU Name", join_unique),
                "Seller Fault Int": ("Seller Fault Int", "max"),
                "Request Cancelled Int": ("Request Cancelled Int", "max"),
                "Has Return Tracking Int": ("Has Return Tracking Int", "max"),
                "Refund Only Int": ("Refund Only Int", "max"),
            }
        )
    )
    pkg["Seller Fault Flag"] = pkg["Seller Fault Int"].eq(1)
    pkg["Return Reason Attribution"] = np.where(pkg["Seller Fault Flag"], "Seller Fault", "No Longer Needed")
    pkg["Request Cancelled Flag"] = pkg["Request Cancelled Int"].eq(1)
    pkg["Has Return Tracking Flag"] = pkg["Has Return Tracking Int"].eq(1)
    pkg["Refund Only Flag"] = pkg["Refund Only Int"].eq(1)
    pkg = pkg.drop(columns=["Seller Fault Int", "Request Cancelled Int", "Has Return Tracking Int", "Refund Only Int"])
    return pkg

def finalize_return_package_segments(pkg: pd.DataFrame, live1_start, live1_end, live2_start, live2_end) -> pd.DataFrame:
    if pkg.empty:
        return pkg
    pkg = pkg.copy()
    pkg["Created Hour"] = pkg["Created Datetime"].dt.hour
    pkg["Live Segment by Created Time"] = pkg["Created Hour"].apply(
        lambda h: classify_live_segment(h, live1_start, live1_end, live2_start, live2_end)
    )
    pkg.loc[pkg["Created Datetime"].isna(), "Live Segment by Created Time"] = "Unknown"
    pkg["Is Live by Created Time"] = pkg["Live Segment by Created Time"].isin(["直播①", "直播②"])
    return pkg


def make_return_segment_summary(return_packages: pd.DataFrame) -> pd.DataFrame:
    total = len(return_packages)
    rows = []
    for seg in ["直播①", "直播②", "直播合计", "非直播", "Unknown"]:
        if seg == "直播合计":
            count = int(return_packages["Live Segment by Created Time"].isin(["直播①", "直播②"]).sum()) if total else 0
        else:
            count = int((return_packages["Live Segment by Created Time"] == seg).sum()) if total else 0
        rows.append([seg, count, pct(count, total)])
    return pd.DataFrame(rows, columns=["Segment", "Returned Packages", "占比 %"])


def make_return_line_breakdown(lines: pd.DataFrame, group_col: str, top_n: int) -> pd.DataFrame:
    if lines.empty or group_col not in lines.columns:
        return pd.DataFrame(columns=[group_col, "Returned Units", "Return Package Count", "SKU Rows", "Pct"])
    total_units = lines["Return Quantity Parsed"].sum()
    out = (
        lines.groupby(group_col, dropna=False)
        .agg(
            **{
                "Returned Units": ("Return Quantity Parsed", "sum"),
                "Return Package Count": ("Return Package Key", "nunique"),
                "SKU Rows": ("Returned SKU Row Count", "sum"),
            }
        )
        .reset_index()
    )
    out[group_col] = out[group_col].apply(lambda x: normalize_text(x, default="Unknown"))
    out["Pct"] = out["Returned Units"].apply(lambda x: pct(x, total_units))
    out = out.sort_values(["Returned Units", "Return Package Count"], ascending=[False, False]).head(top_n).reset_index(drop=True)
    return out


def make_return_package_count_table(pkg: pd.DataFrame, col: str, top_n=10) -> pd.DataFrame:
    if pkg.empty or col not in pkg.columns:
        return pd.DataFrame(columns=[col, "Returned Packages", "Pct"])
    total = len(pkg)
    out = pkg[col].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"}).value_counts().head(top_n).reset_index()
    out.columns = [col, "Returned Packages"]
    out["Pct"] = out["Returned Packages"].apply(lambda x: pct(x, total))
    return out


def make_return_boolean_summary(return_packages: pd.DataFrame) -> pd.DataFrame:
    total = len(return_packages)
    rows = [
        ["Seller Fault", int(return_packages["Seller Fault Flag"].sum()) if total else 0, "Return Reason 不是 No Longer Needed"],
        ["Request Cancelled", int(return_packages["Request Cancelled Flag"].sum()) if total else 0, "S column Return Sub Status"],
        ["已寄出退回包裹", int(return_packages["Has Return Tracking Flag"].sum()) if total else 0, "Q column Return Logistics Tracking ID 有记录"],
        ["Refund Only", int(return_packages["Refund Only Flag"].sum()) if total else 0, "L column Return Type = Refund Only"],
    ]
    out = pd.DataFrame(rows, columns=["Metric", "Returned Packages", "Definition"])
    out["占比 %"] = out["Returned Packages"].apply(lambda x: pct(x, total))
    return out


def make_return_segment_rows(return_summary: pd.DataFrame, color="var(--purple)"):
    if return_summary is None or return_summary.empty:
        return '<div class="empty-note">暂无 returned package 数据</div>'
    max_count = return_summary["Returned Packages"].max()
    rows = []
    for _, r in return_summary.iterrows():
        name = str(r["Segment"])
        width = pct(r["Returned Packages"], max_count, 0) if max_count else 0
        rows.append(f'''
          <div class="break-row">
            <div class="break-name" title="{html.escape(name)}">{html.escape(name)}</div>
            <div class="break-bar-wrap"><div class="break-bar" style="width:{width}%;background:{color}"></div></div>
            <div class="break-num">{fmt_num(r['Returned Packages'])}</div>
            <div class="break-pct">{float(r['占比 %']):.1f}%</div>
          </div>
        ''')
    return "\n".join(rows)


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


    if ctx.get("has_return_data"):
        return_section_html = f'''
  <div class="section">
    <div class="section-label"><span>07</span>Returned 包裹直播归因 · Based on Order Created Time</div>
    <div class="stat-grid">
      <div class="stat green"><div class="stat-lbl">Returned Packages</div><div class="stat-val green">{fmt_num(ctx['return_total_packages'])}</div><div class="stat-sub">按 Tracking / Return Order ID 去重</div></div>
      <div class="stat purple"><div class="stat-lbl">直播时间 Created</div><div class="stat-val purple">{fmt_num(ctx['return_live_packages'])}</div><div class="stat-sub">占 returned packages {fmt_pct(ctx['return_live_packages'], ctx['return_total_packages'])}</div></div>
      <div class="stat blue"><div class="stat-lbl">非直播 Created</div><div class="stat-val blue">{fmt_num(ctx['return_nonlive_packages'])}</div><div class="stat-sub">Created Time 不在直播时段</div></div>
      <div class="stat amber"><div class="stat-lbl">Created Time Unknown</div><div class="stat-val amber">{fmt_num(ctx['return_unknown_packages'])}</div><div class="stat-sub">未能从订单总表匹配</div></div>
    </div>
    <div class="full-panel" style="margin-top:16px">
      <div class="panel-title">Returned package segment breakdown <span class="badge y">Return 表全部视作 returned</span></div>
      {make_return_segment_rows(ctx.get('return_segment_summary'), 'var(--purple)')}
    </div>
  </div>

  <div class="section">
    <div class="section-label"><span>08</span>Returned Orders · 核心指标</div>
    <div class="stat-grid">
      <div class="stat red"><div class="stat-lbl">Seller Fault</div><div class="stat-val red">{fmt_num(ctx['return_seller_fault_packages'])}</div><div class="stat-sub">Return Reason ≠ No Longer Needed · {fmt_pct(ctx['return_seller_fault_packages'], ctx['return_total_packages'])}</div></div>
      <div class="stat amber"><div class="stat-lbl">Request Cancelled</div><div class="stat-val amber">{fmt_num(ctx['return_request_cancelled_packages'])}</div><div class="stat-sub">S column · {fmt_pct(ctx['return_request_cancelled_packages'], ctx['return_total_packages'])}</div></div>
      <div class="stat green"><div class="stat-lbl">已寄出退回包裹</div><div class="stat-val green">{fmt_num(ctx['return_shipped_packages'])}</div><div class="stat-sub">Q column tracking 有记录 · {fmt_pct(ctx['return_shipped_packages'], ctx['return_total_packages'])}</div></div>
      <div class="stat blue"><div class="stat-lbl">Refund Only</div><div class="stat-val blue">{fmt_num(ctx['return_refund_only_packages'])}</div><div class="stat-sub">L column Return Type · {fmt_pct(ctx['return_refund_only_packages'], ctx['return_total_packages'])}</div></div>
    </div>
    <div class="two-col" style="margin-top:16px">
      <div class="panel">
        <div class="panel-title">Return Reason 占比 <span class="badge r">N column</span></div>
        {make_reason_rows(ctx.get('return_reason_df').rename(columns={'Returned Packages':'Order Count'}), 'var(--live1)')}
      </div>
      <div class="panel">
        <div class="panel-title">Seller Fault 归因 <span class="badge y">非 No Longer Needed</span></div>
        {make_reason_rows(ctx.get('return_fault_df').rename(columns={'Returned Packages':'Order Count'}), 'var(--live2)')}
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-label"><span>09</span>Returned Orders · SKU / 款式 Breakdown</div>
    <div class="two-col">
      <div class="panel">
        <div class="panel-title">退货 SKU Top10 <span class="badge b">H column Seller SKU → 产品图册款式英文名</span></div>
        {make_breakdown_rows(ctx['return_sku_breakdown'], 'Return Style English (Catalog)', 'Returned Units', 'var(--nonlive)')}
      </div>
      <div class="panel">
        <div class="panel-title">退货款式 <span class="badge r">J column SKU Name</span></div>
        {make_breakdown_rows(ctx['return_style_breakdown'], 'Return Style (J SKU Name)', 'Returned Units', 'var(--live1)')}
      </div>
    </div>
    <div class="two-col" style="margin-top:16px">
      <div class="panel">
        <div class="panel-title">Return Type <span class="badge y">L column</span></div>
        {make_reason_rows(ctx.get('return_type_df').rename(columns={'Returned Packages':'Order Count'}), 'var(--live2)')}
      </div>
      <div class="panel">
        <div class="panel-title">Return Sub Status <span class="badge b">S column</span></div>
        {make_reason_rows(ctx.get('return_status_df').rename(columns={'Returned Packages':'Order Count'}), 'var(--nonlive)')}
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-label"><span>10</span>Returned Orders · 产品链接 Breakdown</div>
    <div class="two-col">
      <div class="panel">
        <div class="panel-title">Top5 高退货产品链接 <span class="badge b">I column Product Name</span></div>
        {make_breakdown_rows(ctx['return_product_link_top5'], 'Return Product Link (I Product Name)', 'Returned Units', 'var(--nonlive)')}
      </div>
      <div class="panel">
        <div class="panel-title">指定产品链接退货占比 <span class="badge r">按 Return Quantity</span></div>
        {make_breakdown_rows(ctx['return_product_link_targets'], 'Product Link', 'Returned Units', 'var(--live1)')}
      </div>
    </div>
  </div>
'''
        insight_section_no = "11"
    else:
        return_section_html = ""
        insight_section_no = "07"

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
  <h1>Cancelled + Returned<br>订单分析报告</h1>
  <p class="header-sub">订单总表清洗 · Cancelled + Returned · Created Time 直播归因 · SKU/款式/原因拆解</p>
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

  {return_section_html}

  <div class="section">
    <div class="section-label"><span>{insight_section_no}</span>关键洞察 Key Insights</div>
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
        summary_rows = [
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
        ]
        if ctx.get("has_return_data"):
            summary_rows.extend([
                ["Returned Order Source", ctx.get("return_file_name", "")],
                ["Returned Packages", ctx.get("return_total_packages", 0)],
                ["Returned Packages Created in Live Time", ctx.get("return_live_packages", 0)],
                ["Returned Live Created Package %", fmt_pct(ctx.get("return_live_packages", 0), ctx.get("return_total_packages", 0))],
                ["Returned Packages Created in Non-live Time", ctx.get("return_nonlive_packages", 0)],
                ["Returned Packages Unknown Created Time", ctx.get("return_unknown_packages", 0)],
            ])
        summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
        summary.to_excel(writer, index=False, sheet_name="Summary")
        all_orders.to_excel(writer, index=False, sheet_name="Order Level All")
        cancel_orders.to_excel(writer, index=False, sheet_name="Cancelled Orders")
        cancel_lines.to_excel(writer, index=False, sheet_name="Cancelled SKU Lines")
        display_table(style_breakdown, drop_sku_row_count=True).to_excel(writer, index=False, sheet_name="Nail Style Breakdown")
        display_table(product_breakdown, drop_sku_row_count=True).to_excel(writer, index=False, sheet_name="Product Link Breakdown")
        display_table(reason_df).to_excel(writer, index=False, sheet_name="Cancel Reasons")
        pd.DataFrame({
            "Hour": list(range(24)),
            "Cancelled Orders Created Time": ctx["created_hour_counts"],
            "Cancelled Orders Cancelled Time": ctx["cancelled_hour_counts"],
        }).to_excel(writer, index=False, sheet_name="Hourly")
        if ctx.get("has_return_data"):
            ctx.get("return_boolean_summary", pd.DataFrame()).to_excel(writer, index=False, sheet_name="Returned KPI Summary")
            display_return_breakdown_table(ctx.get("return_reason_df", pd.DataFrame())).to_excel(writer, index=False, sheet_name="Return Reasons")
            display_return_breakdown_table(ctx.get("return_sku_breakdown", pd.DataFrame())).to_excel(writer, index=False, sheet_name="Return SKU Top10")
            display_return_breakdown_table(ctx.get("return_style_breakdown", pd.DataFrame())).to_excel(writer, index=False, sheet_name="Return Style J")
            display_return_breakdown_table(ctx.get("return_product_link_top5", pd.DataFrame())).to_excel(writer, index=False, sheet_name="Return Product Top5")
            display_return_breakdown_table(ctx.get("return_product_link_targets", pd.DataFrame())).to_excel(writer, index=False, sheet_name="Return Product Targets")
            display_return_breakdown_table(ctx.get("return_type_df", pd.DataFrame())).to_excel(writer, index=False, sheet_name="Return Type")
            display_return_breakdown_table(ctx.get("return_status_df", pd.DataFrame())).to_excel(writer, index=False, sheet_name="Return Sub Status")
            ctx.get("return_segment_summary", pd.DataFrame()).to_excel(writer, index=False, sheet_name="Returned Segment")
            ctx.get("return_packages_df", pd.DataFrame()).to_excel(writer, index=False, sheet_name="Returned Packages")
            ctx.get("return_lines_df", pd.DataFrame()).to_excel(writer, index=False, sheet_name="Returned SKU Lines")

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
st.title("💅 Cancelled + Returned Orders Report Generator")
st.caption("上传 TikTok Shop 订单总表，程序会自动清洗 Cancelled Orders；可额外上传 Returned Order 表，生成 Cancelled + Returned 综合分析。")

with st.expander("这版程序的核心口径", expanded=True):
    st.markdown(
        """
- **上传文件**：订单总表，不再上传 cancelled order 表。
- **Cancelled 识别**：B column `Order Status` 等于 `Cancelled` 或 `Canceled`。
- **订单口径**：一个 `Order ID` 只算一个订单；一个 `Order ID` 只算一个 cancelled order。
- **直播归因**：使用 **AB column `Created Time`** 判断是否属于直播间购买，而不是 `Cancelled Time`。
- **取消峰值**：使用 `Cancelled Time` 单独分析“顾客什么时候取消”，但它不用于直播归因。
- **甲型 / 产品链接拆解**：基于 cancelled orders 对应的 SKU 行做统计，因为一个订单可能有多个 SKU。
- **Returned 包裹直播归因**：可额外上传 Returned Order 表；该表不看 Order Status，全部视作 returned，并通过 Order ID 回匹配订单总表里的 Created Time。
- **Returned 额外指标**：H column Seller SKU 去掉 `-S/-M/-L` 后通过产品图册匹配款式英文名；J column `SKU Name` 统计退货款式；I column `Product Name` 统计退货产品链接；N column `Return Reason` 统计原因；非 `No longer needed` 全部归为 Seller Fault；S/Q/L column 分别统计 Request Cancelled、已寄出退回包裹、Refund Only。
- **Returned SKU Top10**：只展示产品图册里的款式英文名，不展示 SKU 编码。
        """
    )

uploaded_file = st.file_uploader("上传订单总表 CSV / Excel", type=["csv", "xlsx", "xls"], key="all_order_file")
returned_file = st.file_uploader(
    "可选：上传 Returned Order 表 CSV / Excel（上传后全部视作 returned，不看 Order Status）",
    type=["csv", "xlsx", "xls"],
    key="returned_order_file",
)
product_catalog_file = st.file_uploader(
    "可选：上传产品图册 CSV / Excel（用于把退货 H column Seller SKU 匹配成款式英文名）",
    type=["csv", "xlsx", "xls"],
    key="product_catalog_file",
)

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

# Optional Returned Order package analysis.
# The returned file itself is SKU-line level and all rows are treated as returned.
# Created Time is matched back from the uploaded all-order table by Order ID.
has_return_data = False
return_lines_df = pd.DataFrame()
return_packages_df = pd.DataFrame()
return_segment_summary = pd.DataFrame()
return_reason_df = pd.DataFrame()
return_sku_breakdown = pd.DataFrame()
return_style_breakdown = pd.DataFrame()
return_product_link_top5 = pd.DataFrame()
return_product_link_targets = pd.DataFrame()
return_type_df = pd.DataFrame()
return_status_df = pd.DataFrame()
return_fault_df = pd.DataFrame()
return_boolean_summary = pd.DataFrame()
return_total_packages = return_live_packages = return_nonlive_packages = return_unknown_packages = 0
return_seller_fault_packages = return_request_cancelled_packages = return_shipped_packages = return_refund_only_packages = 0
return_file_name = ""

if returned_file is not None:
    try:
        return_raw = read_uploaded_file(returned_file)
        return_file_name = returned_file.name
        catalog_map = {}
        if product_catalog_file is not None:
            catalog_raw = read_uploaded_file(product_catalog_file)
            catalog_map = build_catalog_sku_name_map(catalog_raw)
            if not catalog_map:
                st.warning("产品图册未识别到 `SKU` 和 `款式英文名称` 两列；退货 SKU Top10 将使用 J column SKU Name 兜底显示。")
        # Use the full all-order file for lookup so returned packages can still match even if
        # the current report date filter is narrower than the uploaded order total table.
        # This lookup only needs Order ID + Created Datetime, so do not rebuild the full
        # order-level report table here; otherwise large files will keep Streamlit running.
        all_orders_lookup_full = (
            lines[["Order ID", "Created Datetime"]]
            .groupby("Order ID", as_index=False, sort=False)["Created Datetime"]
            .min()
        )
        return_lines_df = prepare_return_line_level(
            return_raw, all_orders_lookup_full, live1_start, live1_end, live2_start, live2_end
        )
        return_lines_df = apply_catalog_style_names(return_lines_df, catalog_map)
        return_packages_df = finalize_return_package_segments(
            build_return_package_level(return_lines_df), live1_start, live1_end, live2_start, live2_end
        )
        return_segment_summary = make_return_segment_summary(return_packages_df)
        return_total_packages = len(return_packages_df)
        return_live_packages = int(return_packages_df["Is Live by Created Time"].sum()) if return_total_packages else 0
        return_unknown_packages = int((return_packages_df["Live Segment by Created Time"] == "Unknown").sum()) if return_total_packages else 0
        return_nonlive_packages = return_total_packages - return_live_packages - return_unknown_packages

        return_reason_df = make_return_package_count_table(return_packages_df, "Return Reason", top_n=top_n)
        return_fault_df = make_return_package_count_table(return_packages_df, "Return Reason Attribution", top_n=top_n)
        return_type_df = make_return_package_count_table(return_packages_df, "Return Type", top_n=top_n)
        return_status_df = make_return_package_count_table(return_packages_df, "Return Sub Status", top_n=top_n)
        return_sku_breakdown = make_return_line_breakdown(return_lines_df, "Return Style English (Catalog)", top_n=10)
        return_style_breakdown = make_return_line_breakdown(return_lines_df, "Return Style (J SKU Name)", top_n=top_n)
        return_product_link_top5 = make_return_line_breakdown(return_lines_df, "Return Product Link (I Product Name)", top_n=5)
        return_product_link_targets = make_return_specific_product_link_share(return_lines_df)
        return_boolean_summary = make_return_boolean_summary(return_packages_df)

        return_seller_fault_packages = int(return_packages_df["Seller Fault Flag"].sum()) if return_total_packages else 0
        return_request_cancelled_packages = int(return_packages_df["Request Cancelled Flag"].sum()) if return_total_packages else 0
        return_shipped_packages = int(return_packages_df["Has Return Tracking Flag"].sum()) if return_total_packages else 0
        return_refund_only_packages = int(return_packages_df["Refund Only Flag"].sum()) if return_total_packages else 0
        has_return_data = True
    except Exception as e:
        st.error(f"Returned Order 表读取/分析失败：{e}")

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
    "has_return_data": has_return_data,
    "return_file_name": return_file_name,
    "return_total_packages": return_total_packages,
    "return_live_packages": return_live_packages,
    "return_nonlive_packages": return_nonlive_packages,
    "return_unknown_packages": return_unknown_packages,
    "return_segment_summary": return_segment_summary,
    "return_reason_df": return_reason_df,
    "return_sku_breakdown": return_sku_breakdown,
    "return_style_breakdown": return_style_breakdown,
    "return_product_link_top5": return_product_link_top5,
    "return_product_link_targets": return_product_link_targets,
    "return_type_df": return_type_df,
    "return_status_df": return_status_df,
    "return_fault_df": return_fault_df,
    "return_boolean_summary": return_boolean_summary,
    "return_seller_fault_packages": return_seller_fault_packages,
    "return_request_cancelled_packages": return_request_cancelled_packages,
    "return_shipped_packages": return_shipped_packages,
    "return_refund_only_packages": return_refund_only_packages,
    "return_packages_df": return_packages_df,
    "return_lines_df": return_lines_df,
}
ctx["insights"] = build_insights(
    all_orders, cancel_orders, cancel_lines, style_breakdown, product_breakdown,
    start_date, end_date, live1_start, live1_end, live2_start, live2_end,
)
if has_return_data:
    ctx["insights"].append(
        f"Returned Order 表内共识别 <strong>{fmt_num(return_total_packages)}</strong> 个 returned package，其中原订单 Created Time 落在直播时段的包裹为 <strong>{fmt_num(return_live_packages)}</strong> 个，占 <strong>{fmt_pct(return_live_packages, return_total_packages)}</strong>。该口径不看 Return 表的 Order Status，Return 表上传的行全部视作 returned。"
    )
    ctx["insights"].append(
        f"Returned packages 中，Seller Fault 口径为 Return Reason 不是 No Longer Needed；当前 Seller Fault 包裹 <strong>{fmt_num(return_seller_fault_packages)}</strong> 个，占 <strong>{fmt_pct(return_seller_fault_packages, return_total_packages)}</strong>。Request Cancelled 为 <strong>{fmt_num(return_request_cancelled_packages)}</strong> 个，占 {fmt_pct(return_request_cancelled_packages, return_total_packages)}；已寄出退回包裹为 <strong>{fmt_num(return_shipped_packages)}</strong> 个，占 {fmt_pct(return_shipped_packages, return_total_packages)}；Refund Only 为 <strong>{fmt_num(return_refund_only_packages)}</strong> 个，占 {fmt_pct(return_refund_only_packages, return_total_packages)}。"
    )
    if not return_product_link_top5.empty:
        top_return_product = str(return_product_link_top5.iloc[0]["Return Product Link (I Product Name)"])
        top_return_product_pct = float(return_product_link_top5.iloc[0]["Pct"])
        ctx["insights"].append(
            f"Returned Orders 的 I column 产品链接中，Top1 高退货链接为 <strong>{html.escape(top_return_product)}</strong>，按 Return Quantity 占 returned units 的 <strong>{top_return_product_pct:.1f}%</strong>。建议和该链接对应的直播话术、尺码展示、PDP 图片预期一起复盘。"
        )

    if return_unknown_packages:
        ctx["insights"].append(
            f"有 <strong>{fmt_num(return_unknown_packages)}</strong> 个 returned package 未能通过 Order ID 在订单总表中匹配到 Created Time，因此暂列为 Unknown。建议确认订单总表和 Return 表是否为同一日期范围，或总表是否包含这些 Order ID。"
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

if has_return_data:
    st.subheader("Returned 包裹直播归因：按原订单 Created Time")
    st.caption("Returned Order 表上传后全部视作 returned；程序按包裹去重，优先用 Return Logistics Tracking ID，缺失时用 Return Order ID / Order ID 兜底。Created Time 从订单总表按 Order ID 匹配。")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Returned Packages", fmt_num(return_total_packages))
    r2.metric("Created in Live Time", fmt_num(return_live_packages), help="原订单 Created Time 落在直播①或直播②")
    r3.metric("Live-created %", fmt_pct(return_live_packages, return_total_packages))
    r4.metric("Unknown Created Time", fmt_num(return_unknown_packages), help="Return 表里的 Order ID 未能在订单总表中匹配到 Created Time")
    st.dataframe(return_segment_summary, use_container_width=True, hide_index=True)

    st.subheader("Returned Orders 核心指标")
    rr1, rr2, rr3, rr4 = st.columns(4)
    rr1.metric("Seller Fault", fmt_num(return_seller_fault_packages), help="Return Reason 只要不是 No Longer Needed，全部归为 Seller Fault")
    rr2.metric("Request Cancelled", fmt_num(return_request_cancelled_packages), help="S column Return Sub Status")
    rr3.metric("已寄出退回包裹", fmt_num(return_shipped_packages), help="Q column Return Logistics Tracking ID 有记录")
    rr4.metric("Refund Only", fmt_num(return_refund_only_packages), help="L column Return Type = Refund Only")
    st.dataframe(return_boolean_summary, use_container_width=True, hide_index=True)

    rt1, rt2, rt3, rt4, rt5, rt6, rt7 = st.tabs(["Return Reason", "Seller Fault", "退货 SKU Top10", "退货款式 J", "退货产品链接 I", "Return Type", "Return Package 明细"])
    with rt1:
        st.caption("N column Return Reason：提交退货的各原因占比，按 returned package 去重统计。")
        st.dataframe(display_return_breakdown_table(return_reason_df), use_container_width=True, hide_index=True)
    with rt2:
        st.caption("除 No Longer Needed 外，其他 Return Reason 全部归为 Seller Fault。")
        st.dataframe(display_return_breakdown_table(return_fault_df), use_container_width=True, hide_index=True)
    with rt3:
        st.caption("H column Seller SKU：先忽略 -S / -M / -L 等尺码后缀，再匹配产品图册 `款式英文名称`；只显示 Top10，不显示 SKU 编码。")
        st.dataframe(display_return_breakdown_table(return_sku_breakdown), use_container_width=True, hide_index=True)
    with rt4:
        st.caption("J column SKU Name：作为退货款式，自动去掉末尾尺码。")
        st.dataframe(display_return_breakdown_table(return_style_breakdown), use_container_width=True, hide_index=True)
    with rt5:
        st.caption("I column Product Name：统计 Top5 高退货产品链接，以及指定产品链接的退货占比。")
        st.markdown("**Top5 高退货产品链接**")
        st.dataframe(display_return_breakdown_table(return_product_link_top5), use_container_width=True, hide_index=True)
        st.markdown("**指定产品链接退货占比**")
        st.dataframe(display_return_breakdown_table(return_product_link_targets), use_container_width=True, hide_index=True)
    with rt6:
        st.caption("L column Return Type；Refund Only 占比已在上方核心指标中单独展示。")
        st.dataframe(display_return_breakdown_table(return_type_df), use_container_width=True, hide_index=True)
        st.caption("S column Return Sub Status；Request Cancelled 占比已在上方核心指标中单独展示。")
        st.dataframe(display_return_breakdown_table(return_status_df), use_container_width=True, hide_index=True)
    with rt7:
        st.dataframe(return_packages_df, use_container_width=True, hide_index=True)

st.subheader("Breakdowns")
tab1, tab2, tab3, tab4 = st.tabs(["Cancel Reasons", "甲型 / SKU", "H Column 产品链接", "订单级 Cancelled 明细"])
with tab1:
    st.dataframe(display_table(reason_df), use_container_width=True, hide_index=True)
with tab2:
    st.dataframe(display_table(style_breakdown, drop_sku_row_count=True), use_container_width=True, hide_index=True)
with tab3:
    st.dataframe(display_table(product_breakdown, drop_sku_row_count=True), use_container_width=True, hide_index=True)
with tab4:
    st.dataframe(cancel_orders, use_container_width=True, hide_index=True)

st.subheader("HTML Report Preview")
components.html(html_report, height=1100, scrolling=True)

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "下载 HTML Report",
        data=html_report.encode("utf-8"),
        file_name=f"cancelled_returned_orders_report_{start_date}_{end_date}.html",
        mime="text/html",
        use_container_width=True,
    )
with d2:
    st.download_button(
        "下载清洗后 Excel",
        data=excel_bytes,
        file_name=f"cancelled_returned_orders_cleaned_{start_date}_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
