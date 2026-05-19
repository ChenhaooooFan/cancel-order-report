import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

APP_VERSION = "COLLECTION_OPTIMIZED_20260515_HTML_REPORT"
CANCELLED = {"cancelled", "canceled"}
SIZE_SUFFIX_RE = re.compile(r"[-_ ]?(XS|S|M|L|XL|XXL|XXXL)$", re.I)

st.set_page_config(
    page_title="Cancelled + Returned + Auction Orders Report Generator",
    page_icon="💅",
    layout="wide",
)

COLLECTION_PATTERNS = [
    (["dreamwear", "dream wear"], "DreamWear Collection", "达人带货"),
    (["top trend"], "TOP TREND Collection", "达人带货"),
    (["buy 4 get 1 free", "buy 4"], "Buy 4 Get 1 Free", "官号视频"),
    (["next gen", "nextgen"], "Next Gen Collection", "直播间"),
    (["square"], "Square Collection", "直播间"),
    (["secret", "final sale"], "Secret + Final Sale Collection", "直播间"),
    (["almond"], "Almond (Shape) Collection", "直播间"),
    (["stiletto"], "Stiletto Collection", "直播间"),
    (["new drop"], "NEW DROP Collection", "直播间"),
    (["spring bloom"], "SPRING BLOOM Collection", "直播间"),
    (["summer shine"], "SUMMER SHINE Collection", "直播间"),
    (["valentine"], "Valentine Collection", "直播间"),
    (["christmas glow", "christmas"], "Christmas Glow Collection", "直播间"),
    (["best seller", "bestseller"], "Best Seller Collection", "直播间"),
    (["organizer binder"], "Organizer Binder", "直播间"),
    (["toolkit", "toolkits", "10 pcs"], "10 PCs TOOLKITS", "直播间"),
]

TARGET_LINKS = [
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


def norm(x, default=""):
    if pd.isna(x):
        return default
    s = str(x).replace("\u3000", " ").strip()
    return s if s else default


def clean_cols(df):
    df = df.copy()
    df.columns = [norm(c).replace("\ufeff", "") for c in df.columns]
    return df


def read_file(f):
    if f is None:
        return None
    name = f.name.lower()
    if name.endswith(".csv"):
        try:
            return clean_cols(pd.read_csv(f, dtype=str, encoding="utf-8-sig"))
        except UnicodeDecodeError:
            f.seek(0)
            return clean_cols(pd.read_csv(f, dtype=str, encoding="latin1"))
    return clean_cols(pd.read_excel(f, dtype=str))


def col(df, *names, idx=None):
    low = {c.lower().strip(): c for c in df.columns}

    for n in names:
        k = n.lower().strip()
        if k in low:
            return low[k]

    for n in names:
        k = n.lower().strip()
        for c in df.columns:
            if k in c.lower().strip():
                return c

    if idx is not None and idx < len(df.columns):
        return df.columns[idx]

    return None


def as_id(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def to_num(s):
    return pd.to_numeric(
        s.astype(str).str.replace(r"[$,% ,]", "", regex=True),
        errors="coerce",
    )


def to_dt(s):
    return pd.to_datetime(s, errors="coerce")


def pct(n, d):
    return 0 if not d else n / d * 100


def fmt_money(x):
    return "$0.00" if pd.isna(x) else f"${x:,.2f}"


def remove_size(s):
    return SIZE_SUFFIX_RE.sub("", norm(s)).strip()


def collection_name(x):
    raw = norm(x, "Unknown")
    low = raw.lower()

    for keys, label, _channel in COLLECTION_PATTERNS:
        if any(k in low for k in keys):
            return label

    m = re.search(r"([^|–—-]{1,80}?collection)", raw, flags=re.I)
    if m:
        return m.group(1).strip()

    return raw[:90]


def collection_channel(name):
    low = norm(name).lower()

    for keys, _label, channel in COLLECTION_PATTERNS:
        if any(k in low for k in keys):
            return channel

    if "collection" in low:
        return "直播间"

    return "其他"


def is_cancelled(s):
    return s.astype(str).str.strip().str.lower().isin(CANCELLED)


def is_live_time(dt, start1, end1, start2, end2):
    if pd.isna(dt):
        return np.nan
    h = dt.hour
    return (start1 <= h <= end1) or (start2 <= h <= end2)


def order_level(df):
    order_col = col(df, "Order ID", idx=0)
    if order_col is None:
        raise ValueError("找不到 Order ID column")

    d = df.copy()
    d[order_col] = as_id(d[order_col])

    status = col(d, "Order Status", idx=1)
    created = col(d, "Created Time", idx=27)
    cancelled_time = col(d, "Cancelled Time", idx=32)
    cancel_reason = col(d, "Cancel Reason", idx=34)
    order_amount = col(d, "Order Amount", idx=25)
    ret_type = col(
        d,
        "Cancelation/Return Type",
        "Cancellation/Return Type",
        idx=3,
    )

    agg = {status: "first"} if status else {}

    for c in [created, cancelled_time, cancel_reason, order_amount, ret_type]:
        if c and c not in agg:
            agg[c] = "first"

    od = d.groupby(order_col, dropna=False).agg(agg).reset_index()

    if created:
        od[created] = to_dt(od[created])
    if cancelled_time:
        od[cancelled_time] = to_dt(od[cancelled_time])
    if order_amount:
        od[order_amount] = to_num(od[order_amount])

    return od, {
        "order": order_col,
        "status": status,
        "created": created,
        "cancelled_time": cancelled_time,
        "cancel_reason": cancel_reason,
        "order_amount": order_amount,
        "return_type": ret_type,
    }


def count_table(df, label_col, value_col=None, denom=None, name="Item", top=None):
    if df is None or df.empty or label_col is None:
        return pd.DataFrame(columns=[name, "数量", "占比 %"])

    base = df.copy()
    base[label_col] = base[label_col].map(lambda x: norm(x, "Unknown"))

    if value_col and value_col in base.columns:
        g = base.groupby(label_col)[value_col].sum(min_count=1).reset_index(name="数量")
    else:
        g = base.groupby(label_col).size().reset_index(name="数量")

    denom = denom if denom is not None else g["数量"].sum()
    g["占比 %"] = g["数量"].apply(lambda x: pct(x, denom))
    g = g.sort_values("数量", ascending=False)

    if top:
        g = g.head(top)

    return g.rename(columns={label_col: name})


def reason_table(df, reason_col, denom=None, name="Reason"):
    return count_table(df, reason_col, denom=denom, name=name)


def build_catalog_map(cat):
    if cat is None or cat.empty:
        return {}

    sku = col(cat, "SKU", idx=0)
    eng = col(cat, "款式英文名称", "English", "Style", idx=2)

    if not sku or not eng:
        return {}

    m = {}

    for _, r in cat[[sku, eng]].dropna(how="all").iterrows():
        key = remove_size(r[sku]).upper()
        val = norm(r[eng])
        if key and val:
            m[key] = val

    return m


def cancelled_context(all_df, start1, end1, start2, end2, metric_mode):
    order_df, oc = order_level(all_df)

    total_orders = order_df[oc["order"]].nunique()

    cancel_order_ids = (
        set(order_df.loc[is_cancelled(order_df[oc["status"]]), oc["order"]])
        if oc["status"]
        else set()
    )

    cancel_orders = len(cancel_order_ids)
    order_col = oc["order"]

    lines = all_df.copy()
    lines[order_col] = as_id(lines[order_col])
    cancel_lines = lines[lines[order_col].isin(cancel_order_ids)].copy()

    product_col = col(lines, "Product Name", idx=7)
    sku_col = col(lines, "Seller SKU", idx=6)
    qty_col = col(lines, "Quantity", idx=10)
    reason_col = oc["cancel_reason"]
    amount_col = oc["order_amount"]
    created_col = oc["created"]
    cancelled_time_col = oc["cancelled_time"]

    if qty_col:
        cancel_lines["__qty"] = to_num(cancel_lines[qty_col]).fillna(1)
    else:
        cancel_lines["__qty"] = 1

    if sku_col:
        cancel_lines["__sku_base"] = cancel_lines[sku_col].map(remove_size)

    if product_col:
        cancel_lines["__collection"] = cancel_lines[product_col].map(collection_name)

    metric = "__qty" if metric_mode == "Quantity" else None

    sku_breakdown = (
        count_table(
            cancel_lines,
            "__sku_base",
            metric,
            denom=cancel_lines[metric].sum() if metric else len(cancel_lines),
            name="甲型 / SKU",
            top=20,
        )
        if sku_col
        else pd.DataFrame()
    )

    product_breakdown = (
        count_table(
            cancel_lines,
            product_col,
            metric,
            denom=cancel_lines[metric].sum() if metric else len(cancel_lines),
            name="产品链接",
            top=20,
        )
        if product_col
        else pd.DataFrame()
    )

    reason_df = (
        reason_table(
            order_df[order_df[order_col].isin(cancel_order_ids)],
            reason_col,
            denom=cancel_orders,
            name="Cancel Reason",
        )
        if reason_col
        else pd.DataFrame()
    )

    collection_df = pd.DataFrame()

    if product_col and cancel_orders:
        tmp = cancel_lines[[order_col, product_col]].copy()
        tmp["Collection"] = tmp[product_col].map(collection_name)
        tmp = tmp[
            tmp["Collection"].str.contains(
                "Collection|BUY 4|TOOLKITS|Organizer",
                case=False,
                na=False,
            )
        ]

        collection_df = (
            tmp.groupby("Collection")[order_col]
            .nunique()
            .reset_index(name="Cancelled 行数")
        )

        collection_df["Cancelled 占比"] = collection_df["Cancelled 行数"].apply(
            lambda x: pct(x, cancel_orders)
        )
        collection_df["类型"] = collection_df["Collection"].map(collection_channel)

        collection_df = collection_df[
            ["Collection", "类型", "Cancelled 行数", "Cancelled 占比"]
        ].sort_values("Cancelled 行数", ascending=False)

    live_summary = pd.DataFrame()

    if created_col:
        tmp = order_df[order_df[order_col].isin(cancel_order_ids)].copy()
        tmp["__live"] = tmp[created_col].apply(
            lambda x: is_live_time(x, start1, end1, start2, end2)
        )

        # total orders per segment (all orders, not just cancelled)
        all_tmp = order_df.copy()
        all_tmp["__live"] = all_tmp[created_col].apply(
            lambda x: is_live_time(x, start1, end1, start2, end2)
        )
        seg_live1_total = int(((all_tmp[created_col].dt.hour >= start1) & (all_tmp[created_col].dt.hour <= end1)).sum())
        seg_live2_total = int(((all_tmp[created_col].dt.hour >= start2) & (all_tmp[created_col].dt.hour <= end2)).sum())
        seg_live_total = int((all_tmp["__live"] == True).sum())
        seg_nonlive_total = int((all_tmp["__live"] == False).sum())

        c_live1 = int(((tmp[created_col].dt.hour >= start1) & (tmp[created_col].dt.hour <= end1)).sum())
        c_live2 = int(((tmp[created_col].dt.hour >= start2) & (tmp[created_col].dt.hour <= end2)).sum())
        c_live = int((tmp["__live"] == True).sum())
        c_nonlive = int((tmp["__live"] == False).sum())

        rows = [
            ["直播①", c_live1, seg_live1_total, pct(c_live1, seg_live1_total), pct(c_live1, cancel_orders)],
            ["直播②", c_live2, seg_live2_total, pct(c_live2, seg_live2_total), pct(c_live2, cancel_orders)],
            ["直播合计", c_live, seg_live_total, pct(c_live, seg_live_total), pct(c_live, cancel_orders)],
            ["非直播", c_nonlive, seg_nonlive_total, pct(c_nonlive, seg_nonlive_total), pct(c_nonlive, cancel_orders)],
        ]

        live_summary = pd.DataFrame(rows, columns=["Segment", "Cancelled Orders", "Total Created Orders in Segment", "Segment Cancel Rate", "% of Cancelled Orders"])

    # ── Live vs Non-live deep comparison (all orders, not just cancelled) ─────
    live_vs_nonlive = {}
    if created_col and amount_col:
        all_od = order_df.copy()
        all_od["__live_seg"] = all_od[created_col].apply(
            lambda x: is_live_time(x, start1, end1, start2, end2)
        )
        # True = live, False = non-live; NaN = unknown (exclude)
        live_orders = all_od[all_od["__live_seg"] == True].copy()
        nonlive_orders = all_od[all_od["__live_seg"] == False].copy()

        # For order quantity distribution we need lines, not order_df
        # Build per-order qty sum from all_df lines
        all_lines = all_df.copy()
        all_lines[order_col] = as_id(all_lines[order_col])
        qty_c = col(all_df, "Quantity", idx=10)
        if qty_c:
            all_lines["__qty_num"] = to_num(all_lines[qty_c]).fillna(1)
            order_qty = all_lines.groupby(order_col)["__qty_num"].sum().reset_index()
            order_qty.columns = [order_col, "__order_qty"]
        else:
            order_qty = pd.DataFrame(columns=[order_col, "__order_qty"])

        def seg_stats(seg_df, label):
            n = len(seg_df)
            if n == 0:
                return {"label": label, "n": 0, "cancel_n": 0, "cancel_rate": 0,
                        "aov": np.nan, "q1": 0, "q2": 0, "q3": 0, "q4p": 0,
                        "aov_q1": np.nan, "aov_q2": np.nan, "aov_q3": np.nan, "aov_q4p": np.nan}
            # cancel
            if oc["status"]:
                c_n = int(is_cancelled(seg_df[oc["status"]]).sum())
            else:
                c_n = 0
            # aov (valid only, excl cancelled)
            valid_seg = seg_df[~is_cancelled(seg_df[oc["status"]])] if oc["status"] else seg_df
            aov_v = float(valid_seg[amount_col].mean()) if not valid_seg.empty and valid_seg[amount_col].notna().any() else np.nan
            # qty distribution
            if not order_qty.empty:
                merged = seg_df[[order_col]].merge(order_qty, on=order_col, how="left")
                merged["__order_qty"] = merged["__order_qty"].fillna(1)
                q1 = int((merged["__order_qty"] == 1).sum())
                q2 = int((merged["__order_qty"] == 2).sum())
                q3 = int((merged["__order_qty"] == 3).sum())
                q4p = int((merged["__order_qty"] >= 4).sum())
                # aov per qty group - join amount back
                if amount_col in seg_df.columns:
                    m2 = seg_df[[order_col, amount_col]].merge(order_qty, on=order_col, how="left")
                    m2["__order_qty"] = m2["__order_qty"].fillna(1)
                    aov_q1 = float(m2.loc[m2["__order_qty"] == 1, amount_col].mean()) if (m2["__order_qty"] == 1).any() else np.nan
                    aov_q2 = float(m2.loc[m2["__order_qty"] == 2, amount_col].mean()) if (m2["__order_qty"] == 2).any() else np.nan
                    aov_q3 = float(m2.loc[m2["__order_qty"] == 3, amount_col].mean()) if (m2["__order_qty"] == 3).any() else np.nan
                    aov_q4p = float(m2.loc[m2["__order_qty"] >= 4, amount_col].mean()) if (m2["__order_qty"] >= 4).any() else np.nan
                else:
                    aov_q1 = aov_q2 = aov_q3 = aov_q4p = np.nan
            else:
                q1 = q2 = q3 = q4p = 0
                aov_q1 = aov_q2 = aov_q3 = aov_q4p = np.nan
            return {"label": label, "n": n, "cancel_n": c_n,
                    "cancel_rate": pct(c_n, n), "aov": aov_v,
                    "q1": q1, "q2": q2, "q3": q3, "q4p": q4p,
                    "aov_q1": aov_q1, "aov_q2": aov_q2, "aov_q3": aov_q3, "aov_q4p": aov_q4p}

        live_stat = seg_stats(live_orders, "直播时段")
        nonlive_stat = seg_stats(nonlive_orders, "非直播时段")
        live_vs_nonlive = {"live": live_stat, "nonlive": nonlive_stat,
                           "start1": start1, "end1": end1, "start2": start2, "end2": end2}
    return {
        "total_orders": total_orders,
        "cancel_orders": cancel_orders,
        "cancel_rate": pct(cancel_orders, total_orders),
        "order_df": order_df,
        "cancel_lines": cancel_lines,
        "order_col": order_col,
        "sku_breakdown": sku_breakdown,
        "product_breakdown": product_breakdown,
        "reason_df": reason_df,
        "collection_df": collection_df,
        "live_summary": live_summary,
        "live_vs_nonlive": live_vs_nonlive,
        "amount_col": amount_col,
        "created_col": created_col,
        "cancelled_time_col": cancelled_time_col,
    }


def returned_context(ret_df, all_ctx, cat_map, start1, end1, start2, end2, metric_mode):
    d = ret_df.copy()

    order_col = col(d, "Order ID", idx=1)
    return_order_col = col(d, "Return Order ID", idx=0)
    track_col = col(d, "Return Logistics Tracking ID", idx=16)
    sku_col = col(d, "Seller SKU", idx=7)
    product_col = col(d, "Product Name", idx=8)
    sku_name_col = col(d, "SKU Name", idx=9)
    return_type_col = col(d, "Return Type", idx=11)
    reason_col = col(d, "Return Reason", idx=13)
    qty_col = col(d, "Return Quantity", idx=15)
    sub_col = col(d, "Return Sub Status", idx=18)

    if order_col:
        d[order_col] = as_id(d[order_col])

    d["__qty"] = to_num(d[qty_col]).fillna(1) if qty_col else 1

    if sku_col:
        d["__sku_base"] = d[sku_col].map(remove_size).str.upper()
    else:
        d["__sku_base"] = "Unknown"

    if sku_name_col:
        fallback_style = d[sku_name_col].map(remove_size)
    else:
        fallback_style = d["__sku_base"]

    d["__style"] = d["__sku_base"].map(cat_map).fillna(fallback_style)

    if product_col:
        d["__collection"] = d[product_col].map(collection_name)
    else:
        d["__collection"] = "Unknown"

    keys = []

    for _, r in d.iterrows():
        key = norm(r.get(track_col)) if track_col else ""

        if not key and return_order_col:
            key = norm(r.get(return_order_col))

        if not key and order_col:
            key = norm(r.get(order_col))

        keys.append(key or f"row_{len(keys)}")

    d["__package_key"] = keys
    pkg = d.drop_duplicates("__package_key").copy()
    returned_packages = len(pkg)

    if order_col and all_ctx.get("created_col"):
        od = all_ctx["order_df"]
        created_map = od.set_index(all_ctx["order_col"])[all_ctx["created_col"]]
        pkg["__created"] = pkg[order_col].map(created_map)
        pkg["__live"] = pkg["__created"].apply(
            lambda x: is_live_time(x, start1, end1, start2, end2)
        )
    else:
        pkg["__live"] = np.nan

    live_created = int((pkg["__live"] == True).sum())

    metric = "__qty" if metric_mode == "Quantity" else None

    sku_top10 = count_table(
        d,
        "__style",
        metric,
        denom=d[metric].sum() if metric else len(d),
        name="款式英文名",
        top=10,
    )

    reason_df = (
        reason_table(d, reason_col, denom=len(d), name="Return Reason")
        if reason_col
        else pd.DataFrame()
    )

    product_top5 = (
        count_table(
            d,
            product_col,
            metric,
            denom=d[metric].sum() if metric else len(d),
            name="产品链接",
            top=5,
        )
        if product_col
        else pd.DataFrame()
    )

    target_rows = []

    if product_col:
        low = d[product_col].astype(str).str.lower()

        for key in TARGET_LINKS:
            mask = low.str.contains(re.escape(key.lower()), na=False)
            n = int(d.loc[mask, metric].sum()) if metric else int(mask.sum())
            target_rows.append(
                [
                    key,
                    n,
                    pct(n, d[metric].sum() if metric else len(d)),
                ]
            )

    target_link_df = pd.DataFrame(
        target_rows,
        columns=["Product Link Keyword", "退货行数", "占比 %"],
    )

    collection_df = pd.DataFrame()

    if product_col:
        tmp = d.copy()
        tmp["Collection"] = tmp[product_col].map(collection_name)

        raw_has_collection = tmp[product_col].astype(str).str.contains(
            "Collection|BUY 4|TOOLKIT|Organizer",
            case=False,
            na=False,
        )

        tmp = tmp[raw_has_collection]

        collection_df = tmp.groupby("Collection").size().reset_index(name="退货行数")
        collection_df["退货占比"] = collection_df["退货行数"].apply(
            lambda x: pct(x, len(d))
        )
        collection_df["类型"] = collection_df["Collection"].map(collection_channel)

        collection_df = collection_df[
            ["Collection", "类型", "退货行数", "退货占比"]
        ].sort_values("退货行数", ascending=False)

    seller_fault_n = 0

    if reason_col:
        seller_fault_n = int(
            (d[reason_col].astype(str).str.strip().str.lower() != "no longer needed").sum()
        )

    request_cancelled_n = (
        int(d[sub_col].astype(str).str.contains("cancel", case=False, na=False).sum())
        if sub_col
        else 0
    )

    shipped_back_n = int(d[track_col].map(norm).ne("").sum()) if track_col else 0

    refund_only_n = (
        int(
            d[return_type_col]
            .astype(str)
            .str.contains("refund only", case=False, na=False)
            .sum()
        )
        if return_type_col
        else 0
    )

    # ── Return amount analysis ─────────────────────────────────────────────────
    # Try to find a refund/return amount column in the returned table
    return_amount_col = col(d, "Refund Total", "Return Amount", "Refund Amount", "Total Refund", idx=None)
    total_return_amount = np.nan
    avg_return_amount = np.nan
    if return_amount_col:
        d["__return_amt"] = to_num(d[return_amount_col])
        total_return_amount = float(d["__return_amt"].sum())
        # per-package average (use pkg dedup)
        pkg_amt = pkg.copy()
        pkg_amt["__return_amt"] = pkg_amt[return_amount_col].pipe(to_num) if return_amount_col in pkg_amt.columns else np.nan
        avg_return_amount = float(pkg_amt["__return_amt"].mean()) if not pkg_amt.empty else np.nan
    else:
        # Fallback: try to get order amount from all_ctx if available
        if all_ctx.get("order_col") and all_ctx.get("amount_col") and order_col:
            od = all_ctx["order_df"]
            amt_map = od.set_index(all_ctx["order_col"])[all_ctx["amount_col"]]
            pkg["__order_amt"] = pkg[order_col].map(amt_map)
            total_return_amount = float(pkg["__order_amt"].sum())
            avg_return_amount = float(pkg["__order_amt"].mean())

    # ── Live vs Non-live return comparison ────────────────────────────────────
    ret_live_vs_nonlive = {}
    if "__live" in pkg.columns:
        live_pkg = pkg[pkg["__live"] == True]
        nonlive_pkg = pkg[pkg["__live"] == False]
        unknown_pkg = pkg[pkg["__live"].isna()]

        def ret_seg_stats(seg_pkg, label):
            n = len(seg_pkg)
            sf = 0
            top_reason = ""
            aov = np.nan
            if reason_col and reason_col in d.columns and order_col:
                seg_oids = set(seg_pkg[order_col].astype(str))
                seg_lines = d[d[order_col].astype(str).isin(seg_oids)]
                if not seg_lines.empty:
                    sf = int((seg_lines[reason_col].astype(str).str.strip().str.lower() != "no longer needed").sum())
                    top_r = seg_lines[reason_col].value_counts()
                    top_reason = str(top_r.index[0]) if not top_r.empty else ""
            # aov from order amount
            if "__order_amt" in seg_pkg.columns:
                aov = float(seg_pkg["__order_amt"].mean()) if not seg_pkg.empty else np.nan
            elif "__return_amt" in seg_pkg.columns:
                aov = float(seg_pkg["__return_amt"].mean()) if not seg_pkg.empty else np.nan
            # qty distribution per return package
            q1 = q2 = q3 = q4p = 0
            if "__qty" in d.columns and order_col:
                seg_oids = set(seg_pkg[order_col].astype(str)) if order_col in seg_pkg.columns else set()
                seg_lines = d[d[order_col].astype(str).isin(seg_oids)] if seg_oids else pd.DataFrame()
                if not seg_lines.empty:
                    oqty = seg_lines.groupby(order_col)["__qty"].sum()
                    q1 = int((oqty == 1).sum())
                    q2 = int((oqty == 2).sum())
                    q3 = int((oqty == 3).sum())
                    q4p = int((oqty >= 4).sum())
            return {"label": label, "n": n, "seller_fault": sf, "top_reason": top_reason,
                    "aov": aov, "q1": q1, "q2": q2, "q3": q3, "q4p": q4p}

        ret_live_vs_nonlive = {
            "live": ret_seg_stats(live_pkg, "直播时段退货"),
            "nonlive": ret_seg_stats(nonlive_pkg, "非直播退货"),
            "unknown": ret_seg_stats(unknown_pkg, "Unknown"),
        }

    return {
        "lines": d,
        "package_df": pkg,
        "returned_packages": returned_packages,
        "live_created": live_created,
        "live_pct": pct(live_created, returned_packages),
        "unknown_created": int(pkg["__live"].isna().sum()),
        "sku_top10": sku_top10,
        "reason_df": reason_df,
        "product_top5": product_top5,
        "target_link_df": target_link_df,
        "collection_df": collection_df,
        "seller_fault_n": seller_fault_n,
        "seller_fault_pct": pct(seller_fault_n, len(d)),
        "request_cancelled_n": request_cancelled_n,
        "request_cancelled_pct": pct(request_cancelled_n, len(d)),
        "shipped_back_n": shipped_back_n,
        "shipped_back_pct": pct(shipped_back_n, len(d)),
        "refund_only_n": refund_only_n,
        "refund_only_pct": pct(refund_only_n, len(d)),
        "total_return_amount": total_return_amount,
        "avg_return_amount": avg_return_amount,
        "ret_live_vs_nonlive": ret_live_vs_nonlive,
        "order_col": order_col,
        "reason_col": reason_col,
    }


def merge_collection_summary(ret_ctx, can_ctx):
    ret = ret_ctx.get("collection_df", pd.DataFrame()) if ret_ctx else pd.DataFrame()
    can = can_ctx.get("collection_df", pd.DataFrame()) if can_ctx else pd.DataFrame()

    names = sorted(set(ret.get("Collection", [])) | set(can.get("Collection", [])))

    rows = []

    for name in names:
        r = ret[ret["Collection"] == name]
        c = can[can["Collection"] == name]
        channel = collection_channel(name)

        rows.append(
            [
                name,
                channel,
                int(r["退货行数"].iloc[0]) if not r.empty else 0,
                float(r["退货占比"].iloc[0]) if not r.empty else 0,
                int(c["Cancelled 行数"].iloc[0]) if not c.empty else 0,
                float(c["Cancelled 占比"].iloc[0]) if not c.empty else 0,
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "链接名称",
            "类型",
            "退货行数",
            "退货占比",
            "Cancelled",
            "Cancel占比",
        ],
    )

    if df.empty:
        return df, pd.DataFrame()

    df["差值 pp"] = df["Cancel占比"] - df["退货占比"]
    df = df.sort_values(["退货行数", "Cancelled"], ascending=False)

    channel_rows = []

    for ch in ["达人带货", "官号视频", "直播间"]:
        sub = df[df["类型"] == ch]
        if not sub.empty:
            channel_rows.append(
                [
                    f"{ch}小计",
                    "",
                    sub["退货行数"].sum(),
                    sub["退货占比"].sum(),
                    sub["Cancelled"].sum(),
                    sub["Cancel占比"].sum(),
                ]
            )

    channel_rows.append(
        [
            "全部 Collection 合计",
            "",
            df["退货行数"].sum(),
            df["退货占比"].sum(),
            df["Cancelled"].sum(),
            df["Cancel占比"].sum(),
        ]
    )

    summary = pd.DataFrame(
        channel_rows,
        columns=[
            "链接名称",
            "类型",
            "退货行数",
            "退货占比",
            "Cancelled",
            "Cancel占比",
        ],
    )

    return df, summary


def collection_insights(comp, summary):
    if comp is None or comp.empty:
        return ["暂无 Collection 数据。"]

    top_ret = comp.sort_values("退货行数", ascending=False).iloc[0]
    top_cancel = comp.sort_values("Cancelled", ascending=False).iloc[0]

    live = (
        summary[summary["链接名称"].str.contains("直播间", na=False)]
        if summary is not None and not summary.empty
        else pd.DataFrame()
    )

    creator = (
        summary[summary["链接名称"].str.contains("达人", na=False)]
        if summary is not None and not summary.empty
        else pd.DataFrame()
    )

    official = (
        summary[summary["链接名称"].str.contains("官号", na=False)]
        if summary is not None and not summary.empty
        else pd.DataFrame()
    )

    notes = []

    notes.append(
        f"1. **{top_ret['链接名称']}** 是退货贡献最高的链接，退货占比 {top_ret['退货占比']:.2f}%，"
        f"Cancelled 占比 {top_ret['Cancel占比']:.2f}%。建议结合该链接的 Return Reason 和高退货 SKU 单独深挖。"
    )

    notes.append(
        f"2. **{top_cancel['链接名称']}** 是 Cancelled 贡献最高的链接，Cancelled 占比 {top_cancel['Cancel占比']:.2f}%，"
        f"退货占比 {top_cancel['退货占比']:.2f}%。如果 Cancel 明显高于退货，优先排查履约时效、备货、价格敏感或预期落差。"
    )

    if not creator.empty:
        r, c = creator["退货占比"].iloc[0], creator["Cancel占比"].iloc[0]
        notes.append(
            f"3. **达人带货链接** 退货占比约 {r:.2f}%，Cancel 占比约 {c:.2f}%。"
            "达人链路更容易出现内容展示与实物预期差，建议审查达人展示内容、买家评价与款式质量一致性。"
        )

    if not official.empty:
        r, c = official["退货占比"].iloc[0], official["Cancel占比"].iloc[0]
        notes.append(
            f"4. **官号视频链接** 退货占比约 {r:.2f}%，Cancel 占比约 {c:.2f}%。"
            "若 Cancel 高于退货，说明用户更多在收货前取消，建议优化促销承诺、发货时效与视频引导。"
        )

    if not live.empty:
        r, c = live["退货占比"].iloc[0], live["Cancel占比"].iloc[0]
        notes.append(
            f"5. **直播间链接** 退货占比约 {r:.2f}%，Cancel 占比约 {c:.2f}%。"
            "直播间更偏冲动购买，建议加强尺码引导、材质展示、佩戴效果解释，降低 No Longer Needed 和预期差。"
        )

    return notes


def auction_context(auc_df, ret_ctx, start1=10, end1=18, start2=19, end2=23):
    order_df, oc = order_level(auc_df)

    order_col = oc["order"]
    status_col = oc["status"]
    ret_type_col = oc["return_type"]
    amount_col = oc["order_amount"]

    total = order_df[order_col].nunique()

    cancelled_ids = (
        set(order_df.loc[is_cancelled(order_df[status_col]), order_col])
        if status_col
        else set()
    )
    cancelled_n = len(cancelled_ids)

    return_ids = set()

    if ret_type_col:
        return_ids |= set(
            order_df.loc[
                order_df[ret_type_col]
                .astype(str)
                .str.contains("return/refund", case=False, na=False),
                order_col,
            ]
        )

    if ret_ctx and ret_ctx.get("order_col"):
        ret_order_col = ret_ctx["order_col"]
        return_ids |= set(ret_ctx["lines"][ret_order_col].dropna().astype(str)) & set(
            order_df[order_col].astype(str)
        )

    return_n = len(return_ids)

    valid = order_df[~order_df[order_col].isin(cancelled_ids)].copy()
    valid_aov = valid[amount_col].mean() if amount_col else np.nan

    ret_valid = order_df[order_df[order_col].isin(return_ids - cancelled_ids)].copy()
    ret_aov = ret_valid[amount_col].mean() if amount_col else np.nan

    lines = auc_df.copy()
    lines[order_col] = as_id(lines[order_col])

    sku_col = col(lines, "Seller SKU", idx=6)
    qty_col = col(lines, "Quantity", idx=10)

    if qty_col:
        lines["__qty"] = to_num(lines[qty_col]).fillna(1)
    else:
        lines["__qty"] = 1

    if sku_col:
        lines["__sku_base"] = lines[sku_col].map(remove_size)

    sku_dist = (
        count_table(
            lines,
            "__sku_base",
            "__qty",
            denom=lines["__qty"].sum(),
            name="Seller SKU Base",
        )
        if sku_col
        else pd.DataFrame()
    )

    auction_return_reason = pd.DataFrame()

    if ret_ctx and ret_ctx.get("order_col") and ret_ctx.get("reason_col"):
        rd = ret_ctx["lines"]
        matched = rd[rd[ret_ctx["order_col"]].isin(set(order_df[order_col]))]
        auction_return_reason = reason_table(
            matched,
            ret_ctx["reason_col"],
            denom=len(matched),
            name="Return Reason",
        )

    cancel_reason = (
        reason_table(
            order_df[order_df[order_col].isin(cancelled_ids)],
            oc["cancel_reason"],
            denom=cancelled_n,
            name="Cancel Reason Clean",
        )
        if oc["cancel_reason"]
        else pd.DataFrame()
    )

    # Cancel SKU distribution
    cancel_lines_auc = lines[lines[order_col].isin(cancelled_ids)].copy()
    cancel_sku_dist = pd.DataFrame()
    if sku_col:
        cancel_sku_dist = count_table(
            cancel_lines_auc, "__sku_base", "__qty",
            denom=cancel_lines_auc["__qty"].sum(),
            name="Seller SKU Base",
        )

    # Cancel product links
    product_col_auc = col(lines, "Product Name", idx=7)
    cancel_product = pd.DataFrame()
    if product_col_auc:
        cancel_product = count_table(
            cancel_lines_auc, product_col_auc, "__qty",
            denom=cancel_lines_auc["__qty"].sum(),
            name="Product Link / Product Name",
        )

    # Auction Return: matched return lines
    auction_ret_lines = pd.DataFrame()
    if ret_ctx and ret_ctx.get("order_col"):
        rd = ret_ctx["lines"]
        matched = rd[rd[ret_ctx["order_col"]].isin(set(order_df[order_col]))]
        auction_ret_lines = matched.copy()

    # Return SKU
    auction_return_sku = pd.DataFrame()
    if not auction_ret_lines.empty:
        style_col_r = "__style" if "__style" in auction_ret_lines.columns else None
        if style_col_r:
            qty_col_r = "__qty" if "__qty" in auction_ret_lines.columns else None
            auction_return_sku = count_table(
                auction_ret_lines, style_col_r, qty_col_r,
                denom=auction_ret_lines[qty_col_r].sum() if qty_col_r else len(auction_ret_lines),
                name="Return Style English (Catalog)",
            )

    # Return Product Link
    auction_return_product = pd.DataFrame()
    if not auction_ret_lines.empty:
        ret_prod_col = col(auction_ret_lines, "Product Name", idx=8)
        if ret_prod_col:
            qty_col_r = "__qty" if "__qty" in auction_ret_lines.columns else None
            auction_return_product = count_table(
                auction_ret_lines, ret_prod_col, qty_col_r,
                denom=auction_ret_lines[qty_col_r].sum() if qty_col_r else len(auction_ret_lines),
                name="Return Product Link (I Product Name)",
            )

    # Auction return core metrics
    auc_ret_seller_fault = 0
    auc_ret_request_cancelled = 0
    auc_ret_shipped_back = 0
    auc_ret_refund_only = 0
    if not auction_ret_lines.empty:
        reason_col_r = ret_ctx.get("reason_col") if ret_ctx else None
        sub_col_r = col(auction_ret_lines, "Return Sub Status", idx=18)
        track_col_r = col(auction_ret_lines, "Return Logistics Tracking ID", idx=16)
        ret_type_col_r = col(auction_ret_lines, "Return Type", idx=11)
        if reason_col_r and reason_col_r in auction_ret_lines.columns:
            auc_ret_seller_fault = int((auction_ret_lines[reason_col_r].astype(str).str.strip().str.lower() != "no longer needed").sum())
        if sub_col_r:
            auc_ret_request_cancelled = int(auction_ret_lines[sub_col_r].astype(str).str.contains("cancel", case=False, na=False).sum())
        if track_col_r:
            auc_ret_shipped_back = int(auction_ret_lines[track_col_r].map(norm).ne("").sum())
        if ret_type_col_r:
            auc_ret_refund_only = int(auction_ret_lines[ret_type_col_r].astype(str).str.contains("refund only", case=False, na=False).sum())

    # ── Auction live vs non-live comparison ───────────────────────────────────
    auc_live_vs_nonlive = {}
    created_col_auc = oc["created"]
    if created_col_auc and amount_col:
        order_df["__live_seg"] = order_df[created_col_auc].apply(
            lambda x: is_live_time(x, start1, end1, start2, end2)
        )
        order_qty_auc = lines.groupby(order_col)["__qty"].sum().reset_index()
        order_qty_auc.columns = [order_col, "__order_qty"]

        def auc_seg_stats(seg_df, label):
            n = len(seg_df)
            if n == 0:
                return {"label": label, "n": 0, "cancel_n": 0, "cancel_rate": 0,
                        "aov": np.nan, "q1": 0, "q2": 0, "q3": 0, "q4p": 0,
                        "aov_q1": np.nan, "aov_q2": np.nan, "aov_q3": np.nan, "aov_q4p": np.nan}
            c_n = int(is_cancelled(seg_df[status_col]).sum()) if status_col else 0
            valid_seg = seg_df[~is_cancelled(seg_df[status_col])] if status_col else seg_df
            aov_v = float(valid_seg[amount_col].mean()) if not valid_seg.empty and valid_seg[amount_col].notna().any() else np.nan
            m = seg_df[[order_col, amount_col]].merge(order_qty_auc, on=order_col, how="left")
            m["__order_qty"] = m["__order_qty"].fillna(1)
            q1 = int((m["__order_qty"] == 1).sum())
            q2 = int((m["__order_qty"] == 2).sum())
            q3 = int((m["__order_qty"] == 3).sum())
            q4p = int((m["__order_qty"] >= 4).sum())
            aov_q1 = float(m.loc[m["__order_qty"] == 1, amount_col].mean()) if q1 else np.nan
            aov_q2 = float(m.loc[m["__order_qty"] == 2, amount_col].mean()) if q2 else np.nan
            aov_q3 = float(m.loc[m["__order_qty"] == 3, amount_col].mean()) if q3 else np.nan
            aov_q4p = float(m.loc[m["__order_qty"] >= 4, amount_col].mean()) if q4p else np.nan
            return {"label": label, "n": n, "cancel_n": c_n, "cancel_rate": pct(c_n, n),
                    "aov": aov_v, "q1": q1, "q2": q2, "q3": q3, "q4p": q4p,
                    "aov_q1": aov_q1, "aov_q2": aov_q2, "aov_q3": aov_q3, "aov_q4p": aov_q4p}

        auc_live_vs_nonlive = {
            "live": auc_seg_stats(order_df[order_df["__live_seg"] == True], "直播时段"),
            "nonlive": auc_seg_stats(order_df[order_df["__live_seg"] == False], "非直播时段"),
        }

    return {
        "total": total,
        "cancelled_n": cancelled_n,
        "return_n": return_n,
        "return_rate": pct(return_n, total),
        "valid_n": total - cancelled_n,
        "valid_aov": valid_aov,
        "return_aov": ret_aov,
        "aov_diff": ret_aov - valid_aov
        if pd.notna(ret_aov) and pd.notna(valid_aov)
        else np.nan,
        "sku_dist": sku_dist,
        "return_reason": auction_return_reason,
        "cancel_reason": cancel_reason,
        "cancel_sku_dist": cancel_sku_dist,
        "cancel_product": cancel_product,
        "return_sku": auction_return_sku,
        "return_product": auction_return_product,
        "ret_seller_fault": auc_ret_seller_fault,
        "ret_request_cancelled": auc_ret_request_cancelled,
        "ret_shipped_back": auc_ret_shipped_back,
        "ret_refund_only": auc_ret_refund_only,
        "ret_total": len(auction_ret_lines) if not auction_ret_lines.empty else 0,
        "live_vs_nonlive": auc_live_vs_nonlive,
    }


def show_metric_row(items):
    cols = st.columns(len(items))

    for c, (label, value, delta) in zip(cols, items):
        c.metric(label, value, delta=delta)


def style_pct_df(df, pct_cols=None):
    if df is None or df.empty:
        return df

    out = df.copy()

    pct_cols = pct_cols or [
        c
        for c in out.columns
        if "占比" in c or c.endswith("%") or "Cancel占比" in c or "退货占比" in c
    ]

    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].apply(
                lambda x: f"{x:.2f}%" if isinstance(x, (int, float, np.floating)) else x
            )

    return out


def excel_bytes(sheets):
    bio = BytesIO()

    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, df in sheets:
            safe = re.sub(r"[\\/*?:\[\]]", " ", name)[:31]
            (df if df is not None else pd.DataFrame()).to_excel(
                writer,
                sheet_name=safe,
                index=False,
            )

    return bio.getvalue()

# ==============================
# HTML Report Export
# ==============================
def html_escape(x):
    import html
    return html.escape("" if x is None else str(x))


def _html_df(df, title=None, subtitle=None, max_rows=None):
    if df is None or getattr(df, "empty", True):
        body = '<div class="empty">暂无数据</div>'
    else:
        out = style_pct_df(df.copy())
        if max_rows:
            out = out.head(max_rows)
        body = out.to_html(index=False, escape=True, classes="data-table", border=0)

    title_html = f'<h2>{html_escape(title)}</h2>' if title else ""
    subtitle_html = f'<p class="subtitle-note">{html_escape(subtitle)}</p>' if subtitle else ""
    return f'<section class="report-section">{title_html}{subtitle_html}{body}</section>'


def _html_summary_table(rows, cols):
    """Render a simple 2-col summary table like the example HTMLs."""
    header = "".join(f"<th>{html_escape(c)}</th>" for c in cols)
    body = ""
    for r in rows:
        cells = "".join(f"<td>{html_escape(str(v))}</td>" for v in r)
        body += f"<tr>{cells}</tr>"
    return f'<table class="data-table"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'


def _grid_cards(items):
    """items: list of (label, value, sub)"""
    cards = ""
    for label, value, sub in items:
        sub_html = f'<div class="sub">{html_escape(sub)}</div>' if sub else ""
        cards += f"""<div class="card">
            <div class="lbl">{html_escape(label)}</div>
            <div class="val">{html_escape(value)}</div>
            {sub_html}
        </div>"""
    return f'<div class="grid">{cards}</div>'


def _report_css():
    return """
  :root {
    --bg: #f7f7f5; --surface: #ffffff; --surface2: #f0f0ed;
    --border: rgba(0,0,0,0.08); --text: #1a1a18; --text-muted: #5a5c63; --text-dim: #9a9ca3;
    --live1: #d44a1e; --live2: #c8840a; --nonlive: #2d5fa8;
    --accent: #d44a1e; --green: #2a9e62; --purple: #7c5cbf;
    --mono: 'DM Mono', monospace; --sans: 'Noto Sans SC', sans-serif; --display: 'Playfair Display', serif;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: var(--sans); font-weight: 300; line-height: 1.6; }
  header { border-bottom: 1px solid var(--border); padding: 48px 60px 40px; position: relative; overflow: hidden; }
  header::before { content: ''; position: absolute; top: -80px; right: -80px; width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(212,74,30,0.08) 0%, transparent 70%); pointer-events: none; }
  .header-tag { font-family: var(--mono); font-size: 11px; color: var(--accent); letter-spacing: .15em; text-transform: uppercase; margin-bottom: 14px; }
  h1 { font-family: var(--display); font-size: 42px; font-weight: 700; line-height: 1.15; color: var(--text); margin-bottom: 10px; }
  .header-sub { font-size: 14px; color: var(--text-muted); }
  .header-meta { position: absolute; top: 48px; right: 60px; text-align: right; font-family: var(--mono); font-size: 11px; color: var(--text-dim); line-height: 2; }
  .header-meta strong { display: block; font-size: 28px; color: var(--text); font-weight: 500; letter-spacing: -.02em; }
  main { padding: 0 60px 80px; }
  .section { margin-top: 52px; animation: fadeUp .5s ease both; }
  .section:nth-child(1){animation-delay:.05s}.section:nth-child(2){animation-delay:.15s}.section:nth-child(3){animation-delay:.25s}
  .section:nth-child(4){animation-delay:.35s}.section:nth-child(5){animation-delay:.45s}.section:nth-child(6){animation-delay:.55s}
  .section:nth-child(7){animation-delay:.65s}.section:nth-child(8){animation-delay:.75s}
  @keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
  .section-label { font-family: var(--mono); font-size: 10px; letter-spacing: .18em; text-transform: uppercase;
    color: var(--text-dim); margin-bottom: 18px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
  .section-label span { color: var(--accent); margin-right: 8px; }
  .stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .stat { background: var(--surface); padding: 24px 22px; position: relative; }
  .stat::after { content:''; position:absolute; bottom:0; left:22px; right:22px; height:2px; border-radius:2px; }
  .stat.c1::after{background:var(--live1)}.stat.c2::after{background:var(--live2)}.stat.c3::after{background:var(--nonlive)}.stat.c4::after{background:var(--green)}.stat.c5::after{background:var(--purple)}
  .stat-lbl { font-family: var(--mono); font-size: 10px; color: var(--text-muted); letter-spacing: .08em; margin-bottom: 10px; }
  .stat-val { font-size: 36px; font-weight: 500; letter-spacing: -.03em; line-height: 1; margin-bottom: 6px; }
  .stat-val.orange{color:var(--live1)}.stat-val.amber{color:var(--live2)}.stat-val.blue{color:#6b9ddb}.stat-val.green{color:var(--green)}.stat-val.purple{color:var(--purple)}
  .stat-sub { font-size: 12px; color: var(--text-dim); }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 24px; }
  .panel-title { font-family: var(--mono); font-size: 11px; color: var(--text-muted); letter-spacing: .1em; margin-bottom: 18px; display: flex; align-items: center; gap: 10px; }
  .badge { background: var(--surface2); border: 1px solid var(--border); border-radius: 4px; padding: 2px 8px; font-size: 10px; color: var(--text-dim); }
  .badge.r{border-color:rgba(212,74,30,.3);color:var(--live1)}.badge.y{border-color:rgba(200,132,10,.3);color:var(--live2)}.badge.g{border-color:rgba(42,158,98,.3);color:var(--green)}
  .mini-stats { display: flex; gap: 8px; margin-top: 14px; }
  .mini-stat { flex:1; background:var(--surface2); border-radius:6px; padding:10px 12px; text-align:center; }
  .mini-stat-lbl { font-size:10px; color:var(--text-dim); margin-bottom:3px; }
  .mini-stat-val { font-size:18px; font-weight:500; color:var(--text); }
  .mini-stat-sub { font-size:10px; color:var(--text-dim); margin-top:2px; }
  .chart-wrap { position:relative; width:100%; }
  .legend { display:flex; gap:16px; margin-bottom:12px; flex-wrap:wrap; }
  .legend-item { display:flex; align-items:center; gap:6px; font-size:11px; color:var(--text-muted); }
  .legend-dot { width:10px; height:10px; border-radius:2px; flex-shrink:0; }
  .reason-list { margin-top:4px; }
  .reason-row { display:flex; align-items:center; gap:10px; padding:7px 0; border-bottom:1px solid var(--border); font-size:12px; }
  .reason-row:last-child{border-bottom:none}
  .reason-name { flex:1; color:var(--text-muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .reason-bar-wrap { width:80px; height:4px; background:var(--surface2); border-radius:2px; overflow:hidden; }
  .reason-bar { height:100%; border-radius:2px; }
  .reason-cnt { font-family:var(--mono); font-size:11px; color:var(--text); min-width:26px; text-align:right; }
  .reason-pct { font-family:var(--mono); font-size:10px; color:var(--text-dim); min-width:38px; text-align:right; }
  .live-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
  .live-panel { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:20px; position:relative; overflow:hidden; }
  .live-panel::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
  .live-panel.s1::before{background:var(--live1)}.live-panel.s2::before{background:var(--live2)}.live-panel.s3::before{background:var(--nonlive)}
  .live-title { font-family:var(--mono); font-size:10px; color:var(--text-dim); letter-spacing:.1em; margin-bottom:4px; }
  .live-count { font-size:30px; font-weight:500; letter-spacing:-.03em; margin-bottom:14px; }
  .live-panel.s1 .live-count{color:var(--live1)}.live-panel.s2 .live-count{color:var(--live2)}.live-panel.s3 .live-count{color:#6b9ddb}
  .full-panel { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:24px; }
  .insights { display:flex; flex-direction:column; gap:10px; }
  .insight { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px 20px; font-size:13px; color:var(--text-muted); line-height:1.7; display:flex; gap:14px; align-items:flex-start; }
  .insight-icon { font-family:var(--mono); font-size:10px; color:var(--accent); letter-spacing:.05em; flex-shrink:0; padding-top:3px; }
  .insight strong { color:var(--text); font-weight:500; }
  .action-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
  .action-card { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:18px; position:relative; overflow:hidden; }
  .action-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
  .action-card.a1::before{background:var(--live1)}.action-card.a2::before{background:var(--live2)}.action-card.a3::before{background:var(--nonlive)}.action-card.a4::before{background:var(--green)}.action-card.a5::before{background:var(--purple)}
  .action-label { font-family:var(--mono); font-size:9px; color:var(--text-dim); letter-spacing:.12em; text-transform:uppercase; margin-bottom:8px; }
  .action-title { font-size:13px; font-weight:500; color:var(--text); margin-bottom:6px; }
  .action-body { font-size:12px; color:var(--text-muted); line-height:1.65; }
  .data-table { width:100%; border-collapse:collapse; font-size:13px; }
  .data-table th,.data-table td { border-bottom:1px solid var(--border); text-align:left; padding:9px 10px; vertical-align:top; }
  .data-table th { background:var(--surface2); color:var(--text-muted); font-family:var(--mono); font-size:11px; font-weight:500; }
  .subtitle-note { font-size:12px; color:var(--text-dim); margin-bottom:14px; }
  .report-section { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:24px; }
  h2 { font-size:16px; font-weight:500; color:var(--text); margin-bottom:6px; }
  .empty { color:var(--text-dim); font-style:italic; padding:12px 0; font-size:13px; }
  footer { border-top:1px solid var(--border); margin:0 60px; padding:24px 0; font-family:var(--mono); font-size:10px; color:var(--text-dim); display:flex; justify-content:space-between; }
  @media(max-width:900px){main,header{padding-left:24px;padding-right:24px} h1{font-size:30px} .stat-grid,.two-col,.three-col,.live-grid,.action-grid{grid-template-columns:1fr} footer{margin:0 24px}}
  @media print{.section{break-inside:avoid}}
"""


def _wrap_html(title, tag_line, sub_line, meta_num, meta_label, body, generated_at, version, extra_scripts=""):
    css = _report_css()
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_escape(title)}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Noto+Sans+SC:wght@300;400;500;700&family=Playfair+Display:wght@700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>{css}</style>
</head>
<body>
<header>
  <div class="header-tag">{html_escape(tag_line)}</div>
  <h1>{html_escape(title)}</h1>
  <p class="header-sub">{html_escape(sub_line)}</p>
  <div class="header-meta">
    <strong>{html_escape(str(meta_num))}</strong>
    {html_escape(meta_label)}<br>
    Generated {html_escape(generated_at)}
  </div>
</header>
<main>
{body}
</main>
<footer>
  <span>NailVesta Internal Report · {html_escape(version)}</span>
  <span>Generated {html_escape(generated_at)}</span>
</footer>
{extra_scripts}
</body>
</html>"""


# ── helpers for visual HTML ────────────────────────────────────────────────────

def _reason_rows_html(df, name_col, cnt_col, pct_col, color="var(--accent)", top=8):
    """Render inline bar reason rows from a DataFrame."""
    if df is None or df.empty:
        return '<div class="empty">暂无数据</div>'
    rows = ""
    df2 = df.head(top).copy()
    max_cnt = df2[cnt_col].max() if not df2.empty else 1
    for _, r in df2.iterrows():
        name = html_escape(str(r[name_col]))
        cnt = int(r[cnt_col]) if pd.notna(r[cnt_col]) else 0
        pct_val = float(r[pct_col]) if pd.notna(r[pct_col]) else 0.0
        bar_w = int(cnt / max_cnt * 100) if max_cnt else 0
        rows += f"""<div class="reason-row">
          <div class="reason-name">{name}</div>
          <div class="reason-bar-wrap"><div class="reason-bar" style="width:{bar_w}%;background:{color}"></div></div>
          <div class="reason-cnt">{cnt}</div>
          <div class="reason-pct">{pct_val:.1f}%</div>
        </div>"""
    return f'<div class="reason-list">{rows}</div>'


def _df_section(df, sec_num, title, subtitle="", top=None, color="var(--accent)"):
    """Render a named section with reason-bar rows if 2 cols match pattern, else data table."""
    if df is None or df.empty:
        return f"""<div class="section">
          <div class="section-label"><span>{sec_num:02d}</span>{title}</div>
          <div class="full-panel"><div class="empty">暂无数据</div></div>
        </div>"""
    df2 = df.head(top) if top else df
    # detect pct/count cols
    cnt_cols = [c for c in df2.columns if c not in ["占比 %"] and df2[c].apply(lambda x: str(x).replace(",","").replace(".","").isdigit() if pd.notna(x) else False).all()]
    pct_cols_found = [c for c in df2.columns if "占比" in c or c.endswith("%")]
    name_col = df2.columns[0]
    if cnt_cols and pct_cols_found and len(df2.columns) <= 4:
        html_body = _reason_rows_html(df2, name_col, cnt_cols[0], pct_cols_found[0], color=color)
    else:
        html_body = style_pct_df(df2).to_html(index=False, escape=True, classes="data-table", border=0)
    sub_html = f'<p class="subtitle-note">{html_escape(subtitle)}</p>' if subtitle else ""
    return f"""<div class="section">
      <div class="section-label"><span>{sec_num:02d}</span>{title}</div>
      <div class="full-panel">{sub_html}{html_body}</div>
    </div>"""


def _insight_block(insights_list, sec_num, label="关键洞察 Key Insights"):
    items = ""
    for i, text in enumerate(insights_list):
        icon = f"// {i+1:02d}" if i < len(insights_list)-1 else "// REC"
        items += f'<div class="insight"><div class="insight-icon">{icon}</div><div>{html_escape(text)}</div></div>'
    return f"""<div class="section">
      <div class="section-label"><span>{sec_num:02d}</span>{label}</div>
      <div class="insights">{items}</div>
    </div>"""


def _action_block(actions, sec_num, label="下一步行动建议 Action Plan"):
    """actions: list of (title, body, color_class a1-a5)"""
    cards = ""
    for title, body, cls in actions:
        cards += f"""<div class="action-card {cls}">
          <div class="action-label">Action</div>
          <div class="action-title">{html_escape(title)}</div>
          <div class="action-body">{html_escape(body)}</div>
        </div>"""
    return f"""<div class="section">
      <div class="section-label"><span>{sec_num:02d}</span>{label}</div>
      <div class="action-grid">{cards}</div>
    </div>"""


def _chart_script(charts):
    """charts: list of (canvas_id, type, labels_js, data_js, colors_js, max_val, step)"""
    blocks = ""
    for cid, ctype, labels, data, colors, max_v, step in charts:
        if colors.startswith("["):
            color_arg = f"backgroundColor:{colors}"
        else:
            color_arg = f"backgroundColor:'{colors}'"
        blocks += f"""
new Chart(document.getElementById('{cid}'), {{
  type: '{ctype}',
  data: {{ labels: {labels}, datasets: [{{ data: {data}, {color_arg}, borderRadius: 3 }}] }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ autoSkip: false, maxRotation: 0, font: {{ size: 9 }}, color: '#9a9ca3' }}, grid: {{ color: 'rgba(0,0,0,0.04)' }} }},
      y: {{ beginAtZero: true, max: {max_v}, ticks: {{ stepSize: {step}, font: {{ size: 10 }}, color: '#9a9ca3' }}, grid: {{ color: 'rgba(0,0,0,0.04)' }} }}
    }}
  }}
}});"""
    return f"<script>\n{blocks}\n</script>"


# ── Individual report builders ─────────────────────────────────────────────────

def _live_vs_nonlive_html(lvn, sec_num, canvas_prefix, scripts_list, context="cancel"):
    """
    Render the full live vs non-live comparison section.
    lvn: dict with keys "live" and "nonlive", each a stat dict.
    scripts_list: list to append Chart.js script strings to.
    context: "cancel" | "return" | "auction"
    Returns HTML string.
    """
    if not lvn:
        return ""

    live = lvn.get("live", {})
    nonlive = lvn.get("nonlive", {})
    if not live or not nonlive:
        return ""

    def _fmt(v, fmt="n"):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if fmt == "$": return f"${v:,.2f}"
        if fmt == "%": return f"{v:.1f}%"
        return f"{int(v):,}"

    def _cmp_badge(live_v, nonlive_v, higher_is_worse=False):
        """Return colored badge comparing live vs nonlive."""
        if live_v is None or nonlive_v is None:
            return ""
        try:
            live_f = float(live_v); nonlive_f = float(nonlive_v)
        except Exception:
            return ""
        if np.isnan(live_f) or np.isnan(nonlive_f): return ""
        diff = live_f - nonlive_f
        if abs(diff) < 0.001: return '<span style="font-size:10px;color:var(--text-dim)">≈ 持平</span>'
        color = "var(--live1)" if (diff > 0) == higher_is_worse else "var(--green)"
        arrow = "▲" if diff > 0 else "▼"
        return f'<span style="font-size:10px;color:{color}">{arrow} vs 非直播</span>'

    # ── Overview comparison table ──────────────────────────────────────────────
    ln, nn = live.get("n", 0), nonlive.get("n", 0)
    lc, nc = live.get("cancel_n", 0), nonlive.get("cancel_n", 0)
    lcr, ncr = live.get("cancel_rate", 0), nonlive.get("cancel_rate", 0)
    laov, naov = live.get("aov", np.nan), nonlive.get("aov", np.nan)

    rows_html = ""
    metrics = [
        ("订单量", _fmt(ln), _fmt(nn), False),
        ("取消单数" if context != "return" else "退货包裹数", _fmt(lc), _fmt(nc), True),
        ("取消率" if context != "return" else "退货率", _fmt(lcr, "%"), _fmt(ncr, "%"), True),
        ("平均 AOV（有效订单）", _fmt(laov, "$"), _fmt(naov, "$"), False),
    ]
    for label, lv, nv, higher_worse in metrics:
        try:
            lf = float(str(lv).replace("$","").replace(",","").replace("%",""))
            nf = float(str(nv).replace("$","").replace(",","").replace("%",""))
        except Exception:
            lf = nf = 0
        badge = _cmp_badge(lf, nf, higher_worse)
        rows_html += f"""<tr>
          <td style="color:var(--text-muted);font-size:12px">{html_escape(label)}</td>
          <td style="font-family:var(--mono);font-size:12px;color:var(--live1)">{lv}</td>
          <td style="font-family:var(--mono);font-size:12px;color:var(--nonlive)">{nv}</td>
          <td style="text-align:right">{badge}</td>
        </tr>"""

    overview_table = f"""<table class="data-table" style="font-size:12px">
      <thead><tr>
        <th>指标</th>
        <th style="color:var(--live1)">🔴 直播时段</th>
        <th style="color:var(--nonlive)">🔵 非直播时段</th>
        <th>对比</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""

    # ── Items-per-order distribution ───────────────────────────────────────────
    lq = [live.get("q1",0), live.get("q2",0), live.get("q3",0), live.get("q4p",0)]
    nq = [nonlive.get("q1",0), nonlive.get("q2",0), nonlive.get("q3",0), nonlive.get("q4p",0)]
    lqtot = sum(lq) or 1
    nqtot = sum(nq) or 1
    qty_labels = ["1件", "2件", "3件", "4件+"]
    qty_colors_live = ["#d44a1e","#e8775a","#f0a08a","#f8c8b8"]
    qty_colors_nonlive = ["#2d5fa8","#5b86c8","#8aacd8","#b8cfe8"]

    # grouped bar chart for qty distribution
    qty_cid = f"{canvas_prefix}QtyChart"
    scripts_list.append(f"""
new Chart(document.getElementById('{qty_cid}'), {{
  type: 'bar',
  data: {{
    labels: {str(qty_labels)},
    datasets: [
      {{ label: '直播时段', data: {str([round(q/lqtot*100,1) for q in lq])},
         backgroundColor: '#d44a1e', borderRadius: 3 }},
      {{ label: '非直播时段', data: {str([round(q/nqtot*100,1) for q in nq])},
         backgroundColor: '#2d5fa8', borderRadius: 3 }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ font: {{ size:10, family:'DM Mono' }}, color:'#5a5c63' }} }} }},
    scales: {{
      x: {{ ticks: {{ font:{{ size:10 }}, color:'#9a9ca3' }}, grid:{{ color:'rgba(0,0,0,0.04)' }} }},
      y: {{ beginAtZero:true, max:100, ticks:{{ callback: v=>v+'%', font:{{size:9}}, color:'#9a9ca3' }},
            grid:{{ color:'rgba(0,0,0,0.04)' }} }}
    }}
  }}
}});""")

    # AOV per qty group comparison
    aov_row = lambda label, lv, nv: f"""<div class="reason-row">
      <div class="reason-name" style="font-size:12px">{html_escape(label)}</div>
      <div style="font-family:var(--mono);font-size:11px;color:var(--live1);min-width:70px;text-align:right">{_fmt(lv,'$')}</div>
      <div style="font-family:var(--mono);font-size:11px;color:var(--nonlive);min-width:70px;text-align:right">{_fmt(nv,'$')}</div>
    </div>"""

    aov_breakdown = f"""<div style="margin-top:16px">
      <div class="panel-title" style="margin-bottom:8px">各件数 AOV 对比
        <span style="color:var(--live1);font-size:10px">■ 直播</span>
        <span style="color:var(--nonlive);font-size:10px;margin-left:8px">■ 非直播</span>
      </div>
      <div class="reason-list">
        {aov_row("1件订单 AOV", live.get("aov_q1"), nonlive.get("aov_q1"))}
        {aov_row("2件订单 AOV", live.get("aov_q2"), nonlive.get("aov_q2"))}
        {aov_row("3件订单 AOV", live.get("aov_q3"), nonlive.get("aov_q3"))}
        {aov_row("4件+ 订单 AOV", live.get("aov_q4p"), nonlive.get("aov_q4p"))}
      </div>
    </div>"""

    qty_pct_rows = ""
    for i, (ql, qnl, label) in enumerate(zip(lq, nq, qty_labels)):
        lp = round(ql/lqtot*100, 1)
        np_ = round(qnl/nqtot*100, 1)
        qty_pct_rows += f"""<div class="reason-row">
          <div class="reason-name" style="font-size:12px">{label}</div>
          <div style="font-family:var(--mono);font-size:11px;color:var(--live1);min-width:40px;text-align:right">{ql}</div>
          <div style="font-size:10px;color:var(--live1);min-width:40px;text-align:right">{lp}%</div>
          <div style="font-family:var(--mono);font-size:11px;color:var(--nonlive);min-width:40px;text-align:right">{qnl}</div>
          <div style="font-size:10px;color:var(--nonlive);min-width:40px;text-align:right">{np_}%</div>
        </div>"""

    qty_table = f"""<div class="reason-list" style="margin-bottom:12px">
      <div class="reason-row" style="font-size:10px;color:var(--text-dim);padding-bottom:4px">
        <div class="reason-name">件数</div>
        <div style="min-width:40px;text-align:right;color:var(--live1)">直播量</div>
        <div style="min-width:40px;text-align:right;color:var(--live1)">占比</div>
        <div style="min-width:40px;text-align:right;color:var(--nonlive)">非直播量</div>
        <div style="min-width:40px;text-align:right;color:var(--nonlive)">占比</div>
      </div>
      {qty_pct_rows}
    </div>"""

    return f"""<div class="section">
  <div class="section-label"><span>{sec_num:02d}</span>直播 vs 非直播深度对比 Live vs Non-live Comparison</div>
  <div class="two-col" style="margin-bottom:16px">
    <div class="panel">
      <div class="panel-title">核心指标对比 <span class="badge r">直播①{lvn.get('start1',10)}–{lvn.get('end1',18)}点 ② {lvn.get('start2',19)}–{lvn.get('end2',23)}点</span></div>
      {overview_table}
    </div>
    <div class="panel">
      <div class="panel-title">件数分布 % · 直播 vs 非直播</div>
      <div class="chart-wrap" style="height:180px"><canvas id="{qty_cid}"></canvas></div>
    </div>
  </div>
  <div class="two-col">
    <div class="panel">
      <div class="panel-title">件数明细对比</div>
      {qty_table}
    </div>
    <div class="panel">
      <div class="panel-title">各件数 AOV 对比</div>
      {aov_breakdown}
    </div>
  </div>
</div>"""

def build_cancelled_html(cancel_ctx, top_n=10):
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    c = cancel_ctx
    total = c["total_orders"]
    cancelled = c["cancel_orders"]
    cancel_rate = c["cancel_rate"]

    live_df = c.get("live_summary", pd.DataFrame())
    live_cancel = 0
    live1_n = live2_n = nonlive_n = 0
    if not live_df.empty and "Segment" in live_df.columns and "Cancelled Orders" in live_df.columns:
        def seg_val(seg):
            r = live_df[live_df["Segment"] == seg]
            return int(r["Cancelled Orders"].iloc[0]) if not r.empty else 0
        live1_n = seg_val("直播①")
        live2_n = seg_val("直播②")
        live_cancel = seg_val("直播合计")
        nonlive_n = seg_val("非直播")

    cancel_lines = c.get("cancel_lines", pd.DataFrame())
    cancelled_units = int(cancel_lines["__qty"].sum()) if not cancel_lines.empty and "__qty" in cancel_lines.columns else cancelled

    nonlive_n_display = cancelled - live_cancel if live_cancel else nonlive_n

    # ── 01 stat grid ──
    sec01 = f"""<div class="section">
      <div class="section-label"><span>01</span>总览 Overview</div>
      <div class="stat-grid">
        <div class="stat c1"><div class="stat-lbl">Total Orders</div><div class="stat-val">{total:,}</div><div class="stat-sub">Order ID 去重</div></div>
        <div class="stat c2"><div class="stat-lbl">Cancelled Orders</div><div class="stat-val orange">{cancelled:,}</div><div class="stat-sub">Cancel Rate {cancel_rate:.1f}%</div></div>
        <div class="stat c3"><div class="stat-lbl">直播时段 Cancel</div><div class="stat-val blue">{live_cancel:,}</div><div class="stat-sub">占 Cancelled {pct(live_cancel,cancelled):.1f}%</div></div>
        <div class="stat c4"><div class="stat-lbl">Cancelled SKU Units</div><div class="stat-val green">{cancelled_units:,}</div><div class="stat-sub">按 Quantity 汇总</div></div>
      </div>
    </div>"""

    # ── 02 live attribution chart ──
    reason_df = c.get("reason_df", pd.DataFrame())
    # hour distribution from cancel_lines if created_col available
    cancel_lines2 = c.get("cancel_lines", pd.DataFrame())
    hour_data = [0]*24
    created_col_name = c.get("created_col")
    if created_col_name and not cancel_lines2.empty and created_col_name in cancel_lines2.columns:
        hrs = pd.to_datetime(cancel_lines2[created_col_name], errors="coerce").dt.hour
        for h in hrs.dropna():
            hour_data[int(h)] += 1

    def bar_color_js(h, s1, e1, s2, e2):
        if s1 <= h <= e1: return "#d44a1e"
        if s2 <= h <= e2: return "#c8840a"
        return "#2d5fa8"

    # We'll use s1=10,e1=18,s2=19,e2=23 as defaults; actual values come via ctx but not stored
    colors_list = [bar_color_js(h, 10, 18, 19, 23) for h in range(24)]
    colors_js = "[" + ",".join(f"'{c2}'" for c2 in colors_list) + "]"
    hour_js = str(hour_data)
    max_h = max(hour_data) if hour_data else 10
    max_h = max_h + 2

    sec02 = f"""<div class="section">
      <div class="section-label"><span>02</span>直播归因 · Hourly Cancel 分布</div>
      <div class="full-panel">
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:var(--live1)"></div>直播① 10–18点</div>
          <div class="legend-item"><div class="legend-dot" style="background:var(--live2)"></div>直播② 19–23点</div>
          <div class="legend-item"><div class="legend-dot" style="background:var(--nonlive)"></div>非直播时段</div>
        </div>
        <div class="chart-wrap" style="height:200px"><canvas id="cancelHour"></canvas></div>
        <div class="mini-stats" style="margin-top:18px">
          <div class="mini-stat"><div class="mini-stat-lbl">直播① Cancel</div><div class="mini-stat-val" style="color:var(--live1)">{live1_n}单</div><div class="mini-stat-sub">{pct(live1_n,cancelled):.1f}%</div></div>
          <div class="mini-stat"><div class="mini-stat-lbl">直播② Cancel</div><div class="mini-stat-val" style="color:var(--live2)">{live2_n}单</div><div class="mini-stat-sub">{pct(live2_n,cancelled):.1f}%</div></div>
          <div class="mini-stat"><div class="mini-stat-lbl">非直播 Cancel</div><div class="mini-stat-val" style="color:#6b9ddb">{nonlive_n_display}单</div><div class="mini-stat-sub">{pct(nonlive_n_display,cancelled):.1f}%</div></div>
        </div>
      </div>
    </div>"""

    # ── 03 cancel reasons + live segment breakdown ──
    reason_html_left = ""
    if reason_df is not None and not reason_df.empty:
        reason_html_left = _reason_rows_html(reason_df, reason_df.columns[0], reason_df.columns[1], reason_df.columns[2], color="var(--live1)", top=8)

    live_seg_html = ""
    if not live_df.empty:
        rows_html = style_pct_df(live_df).to_html(index=False, escape=True, classes="data-table", border=0)
        live_seg_html = rows_html

    sec03 = f"""<div class="section">
      <div class="section-label"><span>03</span>取消原因 · Cancel Reasons</div>
      <div class="two-col">
        <div class="panel">
          <div class="panel-title">Cancel Reason 分布 <span class="badge r">Top {min(8,len(reason_df) if reason_df is not None and not reason_df.empty else 0)}</span></div>
          {reason_html_left if reason_html_left else '<div class="empty">暂无数据</div>'}
        </div>
        <div class="panel">
          <div class="panel-title">直播归因明细</div>
          {live_seg_html if live_seg_html else '<div class="empty">暂无数据</div>'}
        </div>
      </div>
    </div>"""

    # ── 04 live segment reason breakdown ──
    sku_df = c.get("sku_breakdown", pd.DataFrame())
    product_df = c.get("product_breakdown", pd.DataFrame())

    sku_html = _reason_rows_html(sku_df, sku_df.columns[0], sku_df.columns[1], sku_df.columns[-1], color="var(--live2)", top=10) if sku_df is not None and not sku_df.empty else '<div class="empty">暂无数据</div>'
    prod_html = _reason_rows_html(product_df, product_df.columns[0], product_df.columns[1], product_df.columns[-1], color="var(--nonlive)", top=10) if product_df is not None and not product_df.empty else '<div class="empty">暂无数据</div>'

    sec04 = f"""<div class="section">
      <div class="section-label"><span>04</span>SKU & 产品链接 Cancel 分布</div>
      <div class="two-col">
        <div class="panel">
          <div class="panel-title">甲型 / Seller SKU</div>
          {sku_html}
        </div>
        <div class="panel">
          <div class="panel-title">产品链接 Top 10</div>
          {prod_html}
        </div>
      </div>
    </div>"""

    # ── 05 collection ──
    coll_df = c.get("collection_df", pd.DataFrame())
    coll_body = style_pct_df(coll_df).to_html(index=False, escape=True, classes="data-table", border=0) if coll_df is not None and not coll_df.empty else '<div class="empty">暂无数据</div>'
    sec05 = f"""<div class="section">
      <div class="section-label"><span>05</span>Collection 链接 Cancel 汇总</div>
      <div class="full-panel">{coll_body}</div>
    </div>"""

    # ── 06 situation analysis & insights ──
    top_reason = reason_df.iloc[0][reason_df.columns[0]] if reason_df is not None and not reason_df.empty else "N/A"
    top_reason_pct = float(reason_df.iloc[0][reason_df.columns[-1]]) if reason_df is not None and not reason_df.empty else 0
    live_pct_of_cancel = pct(live_cancel, cancelled)

    insights_list = [
        f"直播时段贡献 {live_cancel} 单 Cancel，占总取消量 {live_pct_of_cancel:.1f}%。直播①（10–18点）和直播②（19–23点）是主要来源，直播间冲动下单后悔效应显著。",
        f"首要取消原因为「{top_reason}」，占 {top_reason_pct:.1f}%，属于主观意愿改变类，说明用户在下单环节存在选择不确定性，需在直播话术和产品页加强决策引导。",
        f"Cancel Rate {cancel_rate:.1f}%。如高于行业参考水位（3%），需重点优化：结账流程体验、尺码选择清晰度、直播产品展示准确性。",
        f"产品链接取消分布不均，建议对取消量最高的链接单独复盘，排查是否存在描述误导、库存问题或价格锚点偏差。",
    ]
    sec06 = _insight_block(insights_list, 6, "形势分析 Situation Analysis")

    # ── 07 action plan ──
    actions = [
        ("优化直播话术 & 选款引导", "在直播高峰时段（10–15点）加入「确认选择」提示，减少 Bought by mistake。对尺码/套数不明确的产品，直播中增加实物对比展示。", "a1"),
        ("结账流程优化", "精简支付页跳转步骤，提前展示支付方式选项，减少「Need to change payment」导致的取消，特别在直播下单高峰时段前测试流程稳定性。", "a2"),
        ("高取消率产品链接复盘", f"对 Cancel 量 Top 3 的产品链接进行专项复盘：检查描述准确性、价格锚点、库存可用性；考虑暂停或优化取消率 >10% 的链接。", "a3"),
        ("非直播时段 Cancel 治理", f"非直播时段取消率（{pct(nonlive_n_display, total):.1f}%）需单独分析：检查搜索流量和达人视频带来的订单质量，优化产品页详情图和 FAQ。", "a4"),
        ("取消率监控机制", "建立日度 Cancel 监控看板，设置 5% 预警阈值；对 Cancel 突增时段自动推送告警，快速定位问题 SKU 或链接。", "a5"),
    ]
    sec07 = _action_block(actions, 7, "下一步行动建议 Action Plan")

    # ── 08 live vs non-live deep comparison ──
    lvn = c.get("live_vs_nonlive", {})
    if lvn:
        lvn["start1"] = lvn.get("start1", 10)
        lvn["end1"] = lvn.get("end1", 18)
        lvn["start2"] = lvn.get("start2", 19)
        lvn["end2"] = lvn.get("end2", 23)
    extra_scripts_list = []
    sec08 = _live_vs_nonlive_html(lvn, 8, "cancel", extra_scripts_list, context="cancel")

    body = sec01 + sec02 + sec03 + sec04 + sec05 + sec06 + sec07 + sec08
    chart_scripts = _chart_script([
        ("cancelHour", "bar", str([f"{h}h" for h in range(24)]), hour_js, colors_js, max_h, max(1, max_h//5))
    ])
    all_scripts = chart_scripts.replace("</script>", "\n" + "\n".join(extra_scripts_list) + "\n</script>") if extra_scripts_list else chart_scripts

    return _wrap_html(
        "取消订单分析报告",
        f"Cancel Order Analytics · {generated_at}",
        "直播归因 · 原因拆解 · SKU 分布 · 直播对比 · 行动建议",
        f"{cancelled:,}", "去重 Cancelled 订单",
        body, generated_at, APP_VERSION, all_scripts,
    )


def build_returned_html(return_ctx, top_n=10):
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    r = return_ctx
    total_pkg = r["returned_packages"]
    seller_fault_n = r["seller_fault_n"]
    refund_only_n = r["refund_only_n"]
    request_cancelled_n = r["request_cancelled_n"]
    shipped_back_n = r["shipped_back_n"]
    live_created = r["live_created"]
    live_pct_val = r["live_pct"]

    # ── 01 stat grid ──
    sec01 = f"""<div class="section">
      <div class="section-label"><span>01</span>总览 Overview</div>
      <div class="stat-grid">
        <div class="stat c1"><div class="stat-lbl">Returned Packages</div><div class="stat-val orange">{total_pkg:,}</div><div class="stat-sub">按 tracking/order 去重</div></div>
        <div class="stat c2"><div class="stat-lbl">Seller Fault</div><div class="stat-val amber">{seller_fault_n:,}</div><div class="stat-sub">占 {r['seller_fault_pct']:.1f}% · Reason ≠ No Longer Needed</div></div>
        <div class="stat c3"><div class="stat-lbl">Refund Only</div><div class="stat-val blue">{refund_only_n:,}</div><div class="stat-sub">L column Return Type</div></div>
        <div class="stat c4"><div class="stat-lbl">Created in Live</div><div class="stat-val green">{live_created:,}</div><div class="stat-sub">{live_pct_val:.1f}% of returns</div></div>
      </div>
    </div>"""

    # ── 02 core metrics donut chart (via doughnut) ──
    core_labels = ["Seller Fault","Request Cancelled","已寄出","Refund Only","Customer Fault"]
    customer_fault = total_pkg - seller_fault_n
    core_data = [seller_fault_n, request_cancelled_n, shipped_back_n, refund_only_n, customer_fault]
    core_colors = ["#d44a1e","#c8840a","#2d5fa8","#7c5cbf","#2a9e62"]
    core_js_data = str(core_data)
    core_js_colors = "[" + ",".join(f"'{x}'" for x in core_colors) + "]"
    core_labels_js = str(core_labels)

    pkg_df = r.get("package_df", pd.DataFrame())
    live1_ret = live2_ret = nonlive_ret = unknown_ret = 0
    if not pkg_df.empty and "__live" in pkg_df.columns:
        live1_ret = int((pkg_df["__live"] == True).sum())
        nonlive_ret = int((pkg_df["__live"] == False).sum())
        unknown_ret = int(pkg_df["__live"].isna().sum())

    sec02 = f"""<div class="section">
      <div class="section-label"><span>02</span>核心指标 Core Metrics</div>
      <div class="two-col">
        <div class="panel">
          <div class="panel-title">退货包裹构成</div>
          <div class="chart-wrap" style="height:220px"><canvas id="retCore"></canvas></div>
          <div class="legend" style="margin-top:12px;justify-content:center">
            {''.join(f'<div class="legend-item"><div class="legend-dot" style="background:{core_colors[i]}"></div>{core_labels[i]}</div>' for i in range(len(core_labels)))}
          </div>
        </div>
        <div class="panel">
          <div class="panel-title">Returned 核心指标</div>
          <div class="reason-list">
            <div class="reason-row"><div class="reason-name">Seller Fault</div><div class="reason-bar-wrap"><div class="reason-bar" style="width:{int(pct(seller_fault_n,total_pkg))}%;background:var(--live1)"></div></div><div class="reason-cnt">{seller_fault_n}</div><div class="reason-pct">{r['seller_fault_pct']:.1f}%</div></div>
            <div class="reason-row"><div class="reason-name">Request Cancelled</div><div class="reason-bar-wrap"><div class="reason-bar" style="width:{int(pct(request_cancelled_n,total_pkg))}%;background:var(--live2)"></div></div><div class="reason-cnt">{request_cancelled_n}</div><div class="reason-pct">{r['request_cancelled_pct']:.1f}%</div></div>
            <div class="reason-row"><div class="reason-name">已寄出退回包裹</div><div class="reason-bar-wrap"><div class="reason-bar" style="width:{int(pct(shipped_back_n,total_pkg))}%;background:var(--nonlive)"></div></div><div class="reason-cnt">{shipped_back_n}</div><div class="reason-pct">{r['shipped_back_pct']:.1f}%</div></div>
            <div class="reason-row"><div class="reason-name">Refund Only</div><div class="reason-bar-wrap"><div class="reason-bar" style="width:{int(pct(refund_only_n,total_pkg))}%;background:var(--purple)"></div></div><div class="reason-cnt">{refund_only_n}</div><div class="reason-pct">{r['refund_only_pct']:.1f}%</div></div>
          </div>
          <div class="mini-stats" style="margin-top:16px">
            <div class="mini-stat"><div class="mini-stat-lbl">直播来源退货</div><div class="mini-stat-val" style="color:var(--live1)">{live1_ret}</div><div class="mini-stat-sub">{pct(live1_ret,total_pkg):.1f}%</div></div>
            <div class="mini-stat"><div class="mini-stat-lbl">非直播</div><div class="mini-stat-val">{nonlive_ret}</div><div class="mini-stat-sub">{pct(nonlive_ret,total_pkg):.1f}%</div></div>
            <div class="mini-stat"><div class="mini-stat-lbl">Unknown</div><div class="mini-stat-val">{unknown_ret}</div><div class="mini-stat-sub">{pct(unknown_ret,total_pkg):.1f}%</div></div>
          </div>
        </div>
      </div>
    </div>"""

    # ── 03 Return Reasons + SKU Top10 ──
    reason_df = r.get("reason_df", pd.DataFrame())
    sku_df = r.get("sku_top10", pd.DataFrame())

    reason_html_body = _reason_rows_html(reason_df, reason_df.columns[0], reason_df.columns[1], reason_df.columns[-1], color="var(--live1)", top=10) if reason_df is not None and not reason_df.empty else '<div class="empty">暂无数据</div>'
    sku_html_body = _reason_rows_html(sku_df, sku_df.columns[0], sku_df.columns[1], sku_df.columns[-1], color="var(--live2)", top=10) if sku_df is not None and not sku_df.empty else '<div class="empty">暂无数据</div>'

    sec03 = f"""<div class="section">
      <div class="section-label"><span>03</span>退货原因 & SKU 分布</div>
      <div class="two-col">
        <div class="panel"><div class="panel-title">Return Reason <span class="badge r">Top 10</span></div>{reason_html_body}</div>
        <div class="panel"><div class="panel-title">退货款式 Top 10</div>{sku_html_body}</div>
      </div>
    </div>"""

    # ── 04 product links ──
    prod5_df = r.get("product_top5", pd.DataFrame())
    target_df = r.get("target_link_df", pd.DataFrame())
    prod5_html = _reason_rows_html(prod5_df, prod5_df.columns[0], prod5_df.columns[1], prod5_df.columns[-1], color="var(--nonlive)", top=5) if prod5_df is not None and not prod5_df.empty else '<div class="empty">暂无数据</div>'
    target_html_body = style_pct_df(target_df).to_html(index=False, escape=True, classes="data-table", border=0) if target_df is not None and not target_df.empty else '<div class="empty">暂无数据</div>'

    sec04 = f"""<div class="section">
      <div class="section-label"><span>04</span>产品链接退货分析</div>
      <div class="two-col">
        <div class="panel"><div class="panel-title">Top5 高退货产品链接</div>{prod5_html}</div>
        <div class="panel"><div class="panel-title">指定链接退货占比</div>{target_html_body}</div>
      </div>
    </div>"""

    # ── 05 situation analysis ──
    top_reason_name = reason_df.iloc[0][reason_df.columns[0]] if reason_df is not None and not reason_df.empty else "N/A"
    top_reason_pct2 = float(reason_df.iloc[0][reason_df.columns[-1]]) if reason_df is not None and not reason_df.empty else 0
    seller_fault_pct = r['seller_fault_pct']

    insights_list = [
        f"Seller Fault 占 {seller_fault_pct:.1f}%（{seller_fault_n} 单），即非「不再需要」类退货，说明产品质量/包装/物流等执行环节存在实质性问题，需立即介入改善。",
        f"最高退货原因为「{top_reason_name}」（{top_reason_pct2:.1f}%）。若为 No Longer Needed，属冲动购买类，需从直播话术和产品期望管理入手；若为品质类，直接追溯 SKU 供应链。",
        f"Refund Only {refund_only_n} 单（{r['refund_only_pct']:.1f}%）——直接退款不退货，说明客户体验极差或物流成本倒挂，需核实是否有批量性问题 SKU。",
        f"已寄出退回包裹 {shipped_back_n} 单（{r['shipped_back_pct']:.1f}%），仓储逆向处理压力较大，关注入库核验和二次销售良品率。",
    ]
    sec05 = _insight_block(insights_list, 5, "形势分析 Situation Analysis")

    # ── 06 action plan ──
    actions = [
        ("Seller Fault 专项治理", f"对 Seller Fault Top3 的 Return Reason（包装损坏/发错货/缺件）各自建立 SOP 改进表，限期 2 周内将 Seller Fault 率降至 30% 以下。", "a1"),
        ("高退货 SKU 供应链复盘", "对退货 Top5 款式与供应商 QC 报告交叉比对，若退货集中在同一批次，启动换货/退货补偿流程并暂停该批次发货。", "a2"),
        ("Refund Only 批量核查", f"筛选所有 Refund Only 订单，判断是否属于同一 SKU / 时段 / 物流渠道；若集中，触发应急处理协议并联系平台申诉。", "a3"),
        ("冲动购买类退货预防", "对 No Longer Needed 高发链接，在直播中增加「7天使用场景」和「真实买家反馈」内容模块，降低用户收货后预期落差。", "a4"),
        ("退货率监控看板", "建立 SKU 级退货率周报，设置 5% 预警线；对超阈值 SKU 自动下架或限流，保护整体账号健康度。", "a5"),
    ]
    sec06 = _action_block(actions, 6, "下一步行动建议 Action Plan")

    # ── 07 return amount analysis ──
    total_return_amount = r.get("total_return_amount", np.nan)
    avg_return_amount = r.get("avg_return_amount", np.nan)
    amt_str = f"${total_return_amount:,.2f}" if pd.notna(total_return_amount) and total_return_amount > 0 else "—（数据列未找到）"
    avg_str = f"${avg_return_amount:,.2f}" if pd.notna(avg_return_amount) and avg_return_amount > 0 else "—（数据列未找到）"

    sec07 = f"""<div class="section">
      <div class="section-label"><span>07</span>退货金额分析 Return Amount</div>
      <div class="stat-grid">
        <div class="stat c1">
          <div class="stat-lbl">TOTAL RETURN AMOUNT</div>
          <div class="stat-val orange" style="font-size:28px">{amt_str}</div>
          <div class="stat-sub">基于订单金额匹配 / Refund 字段</div>
        </div>
        <div class="stat c2">
          <div class="stat-lbl">AVG RETURN AMOUNT / ORDER</div>
          <div class="stat-val amber" style="font-size:28px">{avg_str}</div>
          <div class="stat-sub">按退货包裹去重后平均</div>
        </div>
        <div class="stat c3">
          <div class="stat-lbl">RETURNED PACKAGES</div>
          <div class="stat-val blue">{total_pkg:,}</div>
          <div class="stat-sub">分母口径</div>
        </div>
        <div class="stat c4">
          <div class="stat-lbl">SELLER FAULT EXPOSURE</div>
          <div class="stat-val green" style="font-size:24px">
            {'${:.2f}'.format(avg_return_amount * seller_fault_n) if pd.notna(avg_return_amount) and avg_return_amount > 0 else '—'}
          </div>
          <div class="stat-sub">Seller Fault {seller_fault_n} 单 × 平均退款额</div>
        </div>
      </div>
    </div>"""

    # ── 08 live vs non-live deep comparison ──
    ret_lvn = r.get("ret_live_vs_nonlive", {})
    extra_scripts_list = []
    sec08 = _live_vs_nonlive_html(ret_lvn, 8, "ret", extra_scripts_list, context="return") if ret_lvn else ""

    body = sec01 + sec02 + sec03 + sec04 + sec05 + sec06 + sec07 + sec08
    donut_script = f"""<script>
new Chart(document.getElementById('retCore'), {{
  type: 'doughnut',
  data: {{
    labels: {core_labels_js},
    datasets: [{{ data: {core_js_data}, backgroundColor: {core_js_colors}, borderWidth: 2, borderColor: '#f7f7f5' }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false, cutout: '60%',
    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: function(c){{ return c.label + ': ' + c.raw }} }} }} }}
  }}
}});
{"".join(extra_scripts_list)}
</script>"""

    return _wrap_html(
        "退货订单分析报告",
        f"Returned Order Analytics · {generated_at}",
        "Seller Fault · Return Amount · Return Reason · 直播对比 · 行动建议",
        f"{total_pkg:,}", "退货包裹总量",
        body, generated_at, APP_VERSION, donut_script,
    )


def build_auction_html(auction_ctx, top_n=10):
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    a = auction_ctx
    total = a["total"]
    cancelled_n = a["cancelled_n"]
    return_n = a["return_n"]
    valid_n = a.get("valid_n", total - cancelled_n)
    valid_aov = a["valid_aov"]
    return_aov = a["return_aov"]
    aov_diff = a.get("aov_diff", np.nan)
    cancel_rate = pct(cancelled_n, total)
    return_rate = a["return_rate"]

    # ── 01 stat grid ──
    sec01 = f"""<div class="section">
      <div class="section-label"><span>01</span>总览 Overview</div>
      <div class="stat-grid">
        <div class="stat c1"><div class="stat-lbl">总 Auction Orders</div><div class="stat-val">{total:,}</div><div class="stat-sub">Order ID 去重</div></div>
        <div class="stat c2"><div class="stat-lbl">Cancelled</div><div class="stat-val orange">{cancelled_n:,}</div><div class="stat-sub">Cancel Rate {cancel_rate:.1f}%</div></div>
        <div class="stat c3"><div class="stat-lbl">申请退货</div><div class="stat-val blue">{return_n:,}</div><div class="stat-sub">退货率 {return_rate:.1f}%</div></div>
        <div class="stat c4"><div class="stat-lbl">有效 Auction AOV</div><div class="stat-val green">{fmt_money(valid_aov)}</div><div class="stat-sub">排除 Cancelled</div></div>
      </div>
    </div>"""

    # ── 02 AOV analysis ──
    aov_diff_display = fmt_money(aov_diff) if pd.notna(aov_diff) else "N/A"
    aov_diff_color = "var(--live1)" if pd.notna(aov_diff) and aov_diff > 0 else "var(--green)"

    sku_df = a.get("sku_dist", pd.DataFrame())
    sku_labels = []
    sku_data = []
    sku_colors_palette = ["#d44a1e","#c8840a","#2d5fa8","#2a9e62","#7c5cbf"]
    if sku_df is not None and not sku_df.empty:
        for _, row in sku_df.iterrows():
            sku_labels.append(str(row[sku_df.columns[0]]))
            sku_data.append(int(row[sku_df.columns[1]]) if pd.notna(row[sku_df.columns[1]]) else 0)
    sku_colors_js = "[" + ",".join(f"'{sku_colors_palette[i % len(sku_colors_palette)]}'" for i in range(len(sku_labels))) + "]"

    sec02 = f"""<div class="section">
      <div class="section-label"><span>02</span>AOV 分析 & SKU 分布</div>
      <div class="two-col">
        <div class="panel">
          <div class="panel-title">Auction AOV 对比</div>
          <div class="mini-stats">
            <div class="mini-stat"><div class="mini-stat-lbl">有效 AOV</div><div class="mini-stat-val" style="color:var(--green)">{fmt_money(valid_aov)}</div><div class="mini-stat-sub">排除 Cancelled</div></div>
            <div class="mini-stat"><div class="mini-stat-lbl">退货 AOV</div><div class="mini-stat-val" style="color:var(--live1)">{fmt_money(return_aov)}</div><div class="mini-stat-sub">已申请退货</div></div>
            <div class="mini-stat"><div class="mini-stat-lbl">AOV 差值</div><div class="mini-stat-val" style="color:{aov_diff_color}">{aov_diff_display}</div><div class="mini-stat-sub">退货 - 有效</div></div>
          </div>
          <div style="margin-top:18px">
            <div class="reason-row"><div class="reason-name">总 Order 数</div><div class="reason-cnt">{total:,}</div><div class="reason-pct"></div></div>
            <div class="reason-row"><div class="reason-name">Cancelled 订单</div><div class="reason-cnt">{cancelled_n:,}</div><div class="reason-pct">{cancel_rate:.1f}%</div></div>
            <div class="reason-row"><div class="reason-name">有效订单（排除 Cancelled）</div><div class="reason-cnt">{valid_n:,}</div><div class="reason-pct"></div></div>
            <div class="reason-row"><div class="reason-name">申请退货</div><div class="reason-cnt">{return_n:,}</div><div class="reason-pct">{return_rate:.1f}%</div></div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-title">Auction Seller SKU 分布</div>
          <div class="chart-wrap" style="height:180px"><canvas id="aucSku"></canvas></div>
          <div class="legend" style="margin-top:12px;justify-content:center">
            {''.join(f'<div class="legend-item"><div class="legend-dot" style="background:{sku_colors_palette[i % len(sku_colors_palette)]}"></div>SKU {sku_labels[i]}</div>' for i in range(len(sku_labels)))}
          </div>
        </div>
      </div>
    </div>"""

    # ── 03 Cancel analysis ──
    cancel_reason_df = a.get("cancel_reason", pd.DataFrame())
    cancel_sku_df = a.get("cancel_sku_dist", pd.DataFrame())
    cancel_product_df = a.get("cancel_product", pd.DataFrame())

    cr_html = _reason_rows_html(cancel_reason_df, cancel_reason_df.columns[0], cancel_reason_df.columns[1], cancel_reason_df.columns[-1], color="var(--live1)", top=8) if cancel_reason_df is not None and not cancel_reason_df.empty else '<div class="empty">暂无数据</div>'
    csku_html = _reason_rows_html(cancel_sku_df, cancel_sku_df.columns[0], cancel_sku_df.columns[1], cancel_sku_df.columns[-1], color="var(--live2)", top=8) if cancel_sku_df is not None and not cancel_sku_df.empty else '<div class="empty">暂无数据</div>'

    cprod_html = '<div class="empty">暂无数据</div>'
    if cancel_product_df is not None and not cancel_product_df.empty:
        cprod_html = style_pct_df(cancel_product_df.head(8)).to_html(index=False, escape=True, classes="data-table", border=0)

    sec03 = f"""<div class="section">
      <div class="section-label"><span>03</span>Auction Cancel 分析</div>
      <div class="two-col">
        <div class="panel"><div class="panel-title">Cancel Reasons</div>{cr_html}</div>
        <div class="panel"><div class="panel-title">Cancel SKU 分布</div>{csku_html}</div>
      </div>
      <div class="full-panel" style="margin-top:16px">
        <div class="panel-title">Cancel 产品链接</div>{cprod_html}
      </div>
    </div>"""

    # ── 04 Return analysis ──
    ret_reason_df = a.get("return_reason", pd.DataFrame())
    ret_sku_df = a.get("return_sku", pd.DataFrame())
    ret_prod_df = a.get("return_product", pd.DataFrame())
    ret_total = a.get("ret_total", 0)
    ret_seller_fault = a.get("ret_seller_fault", 0)
    ret_req_cancel = a.get("ret_request_cancelled", 0)
    ret_shipped = a.get("ret_shipped_back", 0)
    ret_refund = a.get("ret_refund_only", 0)

    rr_html = _reason_rows_html(ret_reason_df, ret_reason_df.columns[0], ret_reason_df.columns[1], ret_reason_df.columns[-1], color="var(--nonlive)", top=8) if ret_reason_df is not None and not ret_reason_df.empty else '<div class="empty">暂无数据</div>'
    rsku_html = _reason_rows_html(ret_sku_df, ret_sku_df.columns[0], ret_sku_df.columns[1], ret_sku_df.columns[-1], color="var(--purple)", top=8) if ret_sku_df is not None and not ret_sku_df.empty else '<div class="empty">暂无数据</div>'

    sec04 = f"""<div class="section">
      <div class="section-label"><span>04</span>Auction Return 分析</div>
      <div class="full-panel" style="margin-bottom:16px">
        <div class="panel-title" style="margin-bottom:14px">Auction Returned 核心指标 <span class="badge">来源：Order ID 对齐 Returned 表</span></div>
        <div class="reason-list">
          <div class="reason-row"><div class="reason-name">Seller Fault（Reason ≠ No Longer Needed）</div><div class="reason-bar-wrap"><div class="reason-bar" style="width:{int(pct(ret_seller_fault,ret_total)) if ret_total else 0}%;background:var(--live1)"></div></div><div class="reason-cnt">{ret_seller_fault}</div><div class="reason-pct">{pct(ret_seller_fault,ret_total):.1f}%</div></div>
          <div class="reason-row"><div class="reason-name">Request Cancelled</div><div class="reason-bar-wrap"><div class="reason-bar" style="width:{int(pct(ret_req_cancel,ret_total)) if ret_total else 0}%;background:var(--live2)"></div></div><div class="reason-cnt">{ret_req_cancel}</div><div class="reason-pct">{pct(ret_req_cancel,ret_total):.1f}%</div></div>
          <div class="reason-row"><div class="reason-name">已寄出退回包裹</div><div class="reason-bar-wrap"><div class="reason-bar" style="width:{int(pct(ret_shipped,ret_total)) if ret_total else 0}%;background:var(--nonlive)"></div></div><div class="reason-cnt">{ret_shipped}</div><div class="reason-pct">{pct(ret_shipped,ret_total):.1f}%</div></div>
          <div class="reason-row"><div class="reason-name">Refund Only</div><div class="reason-bar-wrap"><div class="reason-bar" style="width:{int(pct(ret_refund,ret_total)) if ret_total else 0}%;background:var(--purple)"></div></div><div class="reason-cnt">{ret_refund}</div><div class="reason-pct">{pct(ret_refund,ret_total):.1f}%</div></div>
        </div>
      </div>
      <div class="two-col">
        <div class="panel"><div class="panel-title">Return Reasons</div>{rr_html}</div>
        <div class="panel"><div class="panel-title">Return SKU / 款式</div>{rsku_html}</div>
      </div>
    </div>"""

    # ── 05 situation analysis ──
    top_cancel_reason = cancel_reason_df.iloc[0][cancel_reason_df.columns[0]] if cancel_reason_df is not None and not cancel_reason_df.empty else "N/A"
    insights_list = [
        f"Auction 订单 Cancel Rate {cancel_rate:.1f}%（{cancelled_n}/{total}单）。Auction 全部取消原因为「{top_cancel_reason}」，需重点排查：Auction 价格是否存在结账障碍、用户支付意愿与出价逻辑是否匹配。",
        f"退货 AOV（{fmt_money(return_aov)}）{'高于' if pd.notna(aov_diff) and aov_diff > 0 else '低于'}有效 AOV（{fmt_money(valid_aov)}）{aov_diff_display}，说明高客单价 Auction 订单的退货倾向{'更高' if pd.notna(aov_diff) and aov_diff > 0 else '不高'}，{'需重视高价 SKU 的产品描述准确性。' if pd.notna(aov_diff) and aov_diff > 0 else '价格非主要退货驱动因素。'}",
        f"Auction 退货率 {return_rate:.1f}%（{return_n}单）。相比普通订单，Auction 用户的竞标行为本身带有较高参与度，退货主要来自产品实物与直播展示的预期落差。",
        f"Seller SKU 分布反映 Auction 主推套数结构，需确认高 Cancel/Return 的 SKU 是否为特定价位的套数组合，优化 Auction 起拍价和展示逻辑。",
    ]
    sec05 = _insight_block(insights_list, 5, "形势分析 Situation Analysis")

    # ── 06 action plan ──
    actions = [
        ("Auction 起拍价 & 竞价机制优化", f"Review Cancel 原因集中的 Auction 场次：若为「Customer overdue to pay」，考虑提高保证金比例或缩短支付窗口；若为「Bought by mistake」，优化竞价确认页 UI。", "a1"),
        ("高退货 Auction SKU 复盘", f"对 Auction Return Top SKU 与对应场次的直播内容比对，确认产品展示（颜色/尺寸/材质）与实物是否一致；对 Seller Fault 类退货立即启动 QC 改进。", "a2"),
        ("AOV 与退货率关联分析", f"筛选 AOV > {fmt_money(valid_aov)} 的 Auction 订单，单独追踪退货率；若高价位退货率显著更高，考虑在竞价结束时追加产品细节确认步骤。", "a3"),
        ("Refund Only 专项处理", f"Refund Only {ret_refund} 单需快速核实：是否为物流丢失/严重质量问题？按类型分流，物流问题向平台申诉，质量问题追溯生产批次。", "a4"),
        ("Auction 场次健康监控", "建立 Auction 场次级 Cancel + Return 追踪表，按场次时间/主推 SKU/起拍价分层分析，识别高风险场次模式并提前干预。", "a5"),
    ]
    sec06 = _action_block(actions, 6, "下一步行动建议 Action Plan")

    # ── 07 live vs non-live deep comparison ──
    auc_lvn = a.get("live_vs_nonlive", {})
    extra_auc_scripts = []
    sec07 = _live_vs_nonlive_html(auc_lvn, 7, "auc", extra_auc_scripts, context="auction") if auc_lvn else ""

    body = sec01 + sec02 + sec03 + sec04 + sec05 + sec06 + sec07

    sku_chart = ""
    if sku_labels:
        sku_chart = f"""
new Chart(document.getElementById('aucSku'), {{
  type: 'doughnut',
  data: {{
    labels: {str(sku_labels)},
    datasets: [{{ data: {str(sku_data)}, backgroundColor: {sku_colors_js}, borderWidth: 2, borderColor: '#f7f7f5' }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false, cutout: '55%',
    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: function(c){{ return 'SKU ' + c.label + ': ' + c.raw + '单' }} }} }} }}
  }}
}});"""

    scripts = f"<script>\n{sku_chart}\n{''.join(extra_auc_scripts)}\n</script>"

    return _wrap_html(
        "Auction 订单分析报告",
        f"Auction Order Analytics · {generated_at}",
        "SKU 分布 · AOV 分析 · Cancel & Return 拆解 · 直播对比 · 行动建议",
        f"{total:,}", "Auction 订单总量",
        body, generated_at, APP_VERSION, scripts,
    )


def build_collection_html(comp_df, channel_summary, insights, action_df):
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    # ── chart: channel bar ──
    ch_labels, ch_ret, ch_cancel = [], [], []
    if channel_summary is not None and not channel_summary.empty:
        for _, row in channel_summary.iterrows():
            if "小计" in str(row.iloc[0]) or "合计" in str(row.iloc[0]):
                ch_labels.append(str(row.iloc[0]))
                ret_v = float(row["退货行数"]) if "退货行数" in channel_summary.columns else 0
                can_v = float(row["Cancelled"]) if "Cancelled" in channel_summary.columns else 0
                ch_ret.append(ret_v)
                ch_cancel.append(can_v)

    chart_sec = ""
    if ch_labels:
        chart_sec = f"""<div class="section">
          <div class="section-label"><span>02</span>渠道 Cancel & Return 对比</div>
          <div class="full-panel">
            <div class="legend">
              <div class="legend-item"><div class="legend-dot" style="background:var(--live1)"></div>退货行数</div>
              <div class="legend-item"><div class="legend-dot" style="background:var(--live2)"></div>Cancelled</div>
            </div>
            <div class="chart-wrap" style="height:200px"><canvas id="chBar"></canvas></div>
          </div>
        </div>"""

    comp_html = style_pct_df(comp_df).to_html(index=False, escape=True, classes="data-table", border=0) if comp_df is not None and not comp_df.empty else '<div class="empty">暂无数据</div>'
    summary_html = style_pct_df(channel_summary).to_html(index=False, escape=True, classes="data-table", border=0) if channel_summary is not None and not channel_summary.empty else '<div class="empty">暂无数据</div>'

    sec01 = f"""<div class="section">
      <div class="section-label"><span>01</span>Collection 明细表</div>
      <div class="full-panel">{comp_html}</div>
    </div>
    <div class="section">
      <div class="section-label"><span>02</span>渠道汇总（达人带货 / 官号视频 / 直播间）</div>
      <div class="full-panel">{summary_html}</div>
    </div>"""

    insights_block = ""
    if insights:
        items = "".join(f'<div class="insight"><div class="insight-icon">// {i+1:02d}</div><div>{html_escape(t)}</div></div>' for i, t in enumerate(insights))
        insights_block = f"""<div class="section">
          <div class="section-label"><span>03</span>关键洞察 Key Insights</div>
          <div class="insights">{items}</div>
        </div>"""

    action_html_body = style_pct_df(action_df).to_html(index=False, escape=True, classes="data-table", border=0) if action_df is not None and not action_df.empty else ""
    action_sec = f"""<div class="section">
      <div class="section-label"><span>04</span>渠道动作建议 Action Plan</div>
      <div class="full-panel">{action_html_body}</div>
    </div>"""

    body = sec01 + chart_sec + insights_block + action_sec

    chart_script = ""
    if ch_labels:
        chart_script = f"""<script>
new Chart(document.getElementById('chBar'), {{
  type: 'bar',
  data: {{
    labels: {str(ch_labels)},
    datasets: [
      {{ label: '退货行数', data: {str(ch_ret)}, backgroundColor: '#d44a1e', borderRadius: 4 }},
      {{ label: 'Cancelled', data: {str(ch_cancel)}, backgroundColor: '#c8840a', borderRadius: 4 }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: {{ color: '#9a9ca3' }}, grid: {{ color: 'rgba(0,0,0,0.04)' }} }}, y: {{ beginAtZero: true, ticks: {{ color: '#9a9ca3' }}, grid: {{ color: 'rgba(0,0,0,0.04)' }} }} }}
  }}
}});
</script>"""

    return _wrap_html(
        "Collection 链接综合分析",
        f"Collection Analysis · {generated_at}",
        "达人带货 · 官号视频 · 直播间 · 退货 & Cancel 双口径",
        "", "Collection 综合报告",
        body, generated_at, APP_VERSION, chart_script,
    )


def build_html_report(cancel_ctx, return_ctx=None, auction_ctx=None, comp_df=None, channel_summary=None, insights=None, action_df=None, top_n=10):
    """Combined all-in-one report. Delegates to individual builders and stitches them together."""
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    import re as _re
    def extract_main(html_str):
        m = _re.search(r"<main>(.*?)</main>", html_str, _re.DOTALL)
        return m.group(1).strip() if m else ""
    def extract_scripts(html_str):
        scripts = _re.findall(r"(<script>.*?</script>)", html_str, _re.DOTALL)
        return "\n".join(scripts)

    parts = []
    all_scripts = []

    def add_report(html_str, label, num):
        content = extract_main(html_str)
        scripts = extract_scripts(html_str)
        parts.append(f"""<div style="border-top:3px solid rgba(212,74,30,0.25);margin-top:60px;padding-top:40px">
          <div style="font-family:var(--mono);font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);margin-bottom:24px">{num:02d} · {label}</div>
          {content}
        </div>""")
        if scripts:
            all_scripts.append(scripts)

    add_report(build_cancelled_html(cancel_ctx, top_n), "Cancelled Orders Report", 1)
    if return_ctx:
        add_report(build_returned_html(return_ctx, top_n), "Returned Orders Report", 2)
    if auction_ctx:
        add_report(build_auction_html(auction_ctx, top_n), "Auction Orders Report", 3)
    if comp_df is not None:
        add_report(build_collection_html(comp_df, channel_summary, insights, action_df), "Collection 综合分析", 4)

    combined_scripts = "\n".join(all_scripts)
    body = "\n".join(parts)

    return _wrap_html(
        "NailVesta 订单综合分析报告",
        f"Full Orders Report · {generated_at}",
        "Cancelled · Returned · Auction · Collection 四大模块",
        "All-in-One", "综合报告",
        body, generated_at, APP_VERSION, combined_scripts,
    )


st.title("💅 Cancelled + Returned + Auction Orders Report Generator")
st.caption("上传 TikTok Shop 订单总表；可额外上传 Returned Order 表、产品图册、Auction 订单表。")
st.success(
    f"✅ 版本确认：{APP_VERSION}｜含 Auction + AOV + Collection 渠道汇总表 + 链路启示分析"
)

with st.expander("这版程序的核心口径", expanded=True):
    st.markdown(
        """
- **Cancelled**：订单总表 B column `Order Status` = `Cancelled/Canceled`；一个 `Order ID` 只算一个 Cancel。
- **Returned**：Returned Order 表上传后全部视为 returned；表是一行一个 SKU。
- **Returned Collection**：读 Returned 表 I column `Product Name`，带 Collection/Buy4/Toolkit/Organizer 的链接按 SKU 行统计，分母是 Returned Excel 总行数。
- **Cancelled Collection**：读 cancelled 对应 SKU 行的 H/Product Name，按 Collection 归类后统计唯一 `Order ID`，分母是总 cancelled orders。
- **直播归因**：用订单总表 AB column `Created Time` 判断，不用 Cancelled Time。
- **Auction**：Auction 表单独统计全表，不受左侧日期筛选；AOV 用 Z column `Order Amount`，排除 cancelled 后计算有效 AOV。
"""
    )

with st.sidebar:
    st.header("Report Settings")

    metric_mode = st.radio(
        "SKU / 产品链接统计口径",
        ["Quantity", "SKU 行数"],
        index=0,
        horizontal=False,
    )

    st.caption("Quantity = 用 Quantity / Return Quantity 汇总；SKU 行数 = 一行算一条。")

    st.subheader("直播时间")

    c1, c2 = st.columns(2)
    live1_start = c1.number_input("直播①开始", 0, 23, 10)
    live1_end = c2.number_input("直播①结束", 0, 23, 18)

    c3, c4 = st.columns(2)
    live2_start = c3.number_input("直播②开始", 0, 23, 19)
    live2_end = c4.number_input("直播②结束", 0, 23, 23)

    top_n = st.slider("Breakdown 显示 Top N", 5, 30, 10)


all_file = st.file_uploader(
    "1）上传订单总表 CSV / Excel",
    type=["csv", "xlsx", "xls"],
    key="all_order",
)

returned_file = st.file_uploader(
    "2）可选：上传 Returned Order 表 CSV / Excel",
    type=["csv", "xlsx", "xls"],
    key="returned_order",
)

product_catalog_file = st.file_uploader(
    "3）可选：上传产品图册 CSV / Excel（用于 Seller SKU 匹配款式英文名）",
    type=["csv", "xlsx", "xls"],
    key="catalog",
)

auction_file = st.file_uploader(
    "4）【NEW】可选：上传 Auction 订单详情表 CSV / Excel",
    type=["csv", "xlsx", "xls"],
    key="auction_file",
)

if not all_file:
    st.info("请先上传 TikTok Shop 导出的订单总表。")
    st.stop()


all_df = read_file(all_file)
cat_df = read_file(product_catalog_file) if product_catalog_file else None
ret_df = read_file(returned_file) if returned_file else None
auction_df = read_file(auction_file) if auction_file else None

cat_map = build_catalog_map(cat_df)

try:
    cancel_ctx = cancelled_context(
        all_df,
        live1_start,
        live1_end,
        live2_start,
        live2_end,
        metric_mode,
    )

    return_ctx = (
        returned_context(
            ret_df,
            cancel_ctx,
            cat_map,
            live1_start,
            live1_end,
            live2_start,
            live2_end,
            metric_mode,
        )
        if ret_df is not None
        else None
    )

    comp_df, channel_summary = merge_collection_summary(return_ctx, cancel_ctx)
    insights = collection_insights(comp_df, channel_summary)

    auction_ctx = (
        auction_context(auction_df, return_ctx, live1_start, live1_end, live2_start, live2_end)
        if auction_df is not None
        else None
    )

except Exception as e:
    st.error(f"程序读取失败：{e}")
    st.stop()


st.markdown("## 报告输出")

tab_names = ["Cancelled Report"]

if return_ctx:
    tab_names.append("Returned Report")

if auction_ctx:
    tab_names.append("Auction Report")

tab_names.append("Collection 汇总")
tab_names.append("Downloads")

tabs = st.tabs(tab_names)

idx = 0

with tabs[idx]:
    st.header("Cancelled Orders 核心结果")

    show_metric_row(
        [
            ("总 Order 数", f"{cancel_ctx['total_orders']:,}", None),
            (
                "Cancelled 订单",
                f"{cancel_ctx['cancel_orders']:,}",
                f"{cancel_ctx['cancel_rate']:.2f}%",
            ),
            ("Cancelled Rate", f"{cancel_ctx['cancel_rate']:.2f}%", None),
        ]
    )

    sub = st.tabs(
        [
            "Cancel Reasons",
            "甲型 / SKU",
            "产品链接",
            "Collection 链接",
            "直播归因",
            "直播 vs 非直播对比",
        ]
    )

    with sub[0]:
        st.dataframe(
            style_pct_df(cancel_ctx["reason_df"].head(top_n)),
            use_container_width=True,
            hide_index=True,
        )

    with sub[1]:
        st.dataframe(
            style_pct_df(cancel_ctx["sku_breakdown"].head(top_n)),
            use_container_width=True,
            hide_index=True,
        )

    with sub[2]:
        st.dataframe(
            style_pct_df(cancel_ctx["product_breakdown"].head(top_n)),
            use_container_width=True,
            hide_index=True,
        )

    with sub[3]:
        st.dataframe(
            style_pct_df(cancel_ctx["collection_df"]),
            use_container_width=True,
            hide_index=True,
        )

    with sub[4]:
        st.dataframe(
            style_pct_df(cancel_ctx["live_summary"]),
            use_container_width=True,
            hide_index=True,
        )

    with sub[5]:
        lvn = cancel_ctx.get("live_vs_nonlive", {})
        if lvn:
            live_s = lvn.get("live", {})
            nonlive_s = lvn.get("nonlive", {})
            st.markdown(f"**直播时段** `{lvn.get('start1',10)}–{lvn.get('end1',18)}点` + `{lvn.get('start2',19)}–{lvn.get('end2',23)}点`")
            comp_rows = [
                ["订单量", live_s.get("n",0), nonlive_s.get("n",0)],
                ["取消单数", live_s.get("cancel_n",0), nonlive_s.get("cancel_n",0)],
                ["取消率", f"{live_s.get('cancel_rate',0):.2f}%", f"{nonlive_s.get('cancel_rate',0):.2f}%"],
                ["平均 AOV（有效）", f"${live_s.get('aov',0):,.2f}" if live_s.get('aov') else "—", f"${nonlive_s.get('aov',0):,.2f}" if nonlive_s.get('aov') else "—"],
                ["1件订单占比", f"{live_s.get('q1',0)}/{live_s.get('n',1)} ({live_s.get('q1',0)/max(live_s.get('n',1),1)*100:.1f}%)", f"{nonlive_s.get('q1',0)}/{nonlive_s.get('n',1)} ({nonlive_s.get('q1',0)/max(nonlive_s.get('n',1),1)*100:.1f}%)"],
                ["2件订单占比", f"{live_s.get('q2',0)}/{live_s.get('n',1)} ({live_s.get('q2',0)/max(live_s.get('n',1),1)*100:.1f}%)", f"{nonlive_s.get('q2',0)}/{nonlive_s.get('n',1)} ({nonlive_s.get('q2',0)/max(nonlive_s.get('n',1),1)*100:.1f}%)"],
                ["3件订单占比", f"{live_s.get('q3',0)}/{live_s.get('n',1)} ({live_s.get('q3',0)/max(live_s.get('n',1),1)*100:.1f}%)", f"{nonlive_s.get('q3',0)}/{nonlive_s.get('n',1)} ({nonlive_s.get('q3',0)/max(nonlive_s.get('n',1),1)*100:.1f}%)"],
                ["4件+订单占比", f"{live_s.get('q4p',0)}/{live_s.get('n',1)} ({live_s.get('q4p',0)/max(live_s.get('n',1),1)*100:.1f}%)", f"{nonlive_s.get('q4p',0)}/{nonlive_s.get('n',1)} ({nonlive_s.get('q4p',0)/max(nonlive_s.get('n',1),1)*100:.1f}%)"],
                ["1件 AOV", f"${live_s.get('aov_q1',0):,.2f}" if live_s.get('aov_q1') else "—", f"${nonlive_s.get('aov_q1',0):,.2f}" if nonlive_s.get('aov_q1') else "—"],
                ["2件 AOV", f"${live_s.get('aov_q2',0):,.2f}" if live_s.get('aov_q2') else "—", f"${nonlive_s.get('aov_q2',0):,.2f}" if nonlive_s.get('aov_q2') else "—"],
                ["3件 AOV", f"${live_s.get('aov_q3',0):,.2f}" if live_s.get('aov_q3') else "—", f"${nonlive_s.get('aov_q3',0):,.2f}" if nonlive_s.get('aov_q3') else "—"],
                ["4件+ AOV", f"${live_s.get('aov_q4p',0):,.2f}" if live_s.get('aov_q4p') else "—", f"${nonlive_s.get('aov_q4p',0):,.2f}" if nonlive_s.get('aov_q4p') else "—"],
            ]
            st.dataframe(pd.DataFrame(comp_rows, columns=["指标", "🔴 直播时段", "🔵 非直播时段"]), use_container_width=True, hide_index=True)
        else:
            st.info("需要订单总表含 Created Time 和 Order Amount 列才能计算。")


if return_ctx:
    with tabs[idx]:
        st.header("Returned Orders 核心结果")

        show_metric_row(
            [
                ("Returned Packages", f"{return_ctx['returned_packages']:,}", None),
                (
                    "Created in Live Time",
                    f"{return_ctx['live_created']:,}",
                    f"{return_ctx['live_pct']:.2f}%",
                ),
                (
                    "Unknown Created Time",
                    f"{return_ctx['unknown_created']:,}",
                    None,
                ),
                (
                    "Seller Fault",
                    f"{return_ctx['seller_fault_n']:,}",
                    f"{return_ctx['seller_fault_pct']:.2f}%",
                ),
            ]
        )

        show_metric_row(
            [
                (
                    "Request Cancelled",
                    f"{return_ctx['request_cancelled_n']:,}",
                    f"{return_ctx['request_cancelled_pct']:.2f}%",
                ),
                (
                    "已寄出退回包裹",
                    f"{return_ctx['shipped_back_n']:,}",
                    f"{return_ctx['shipped_back_pct']:.2f}%",
                ),
                (
                    "Refund Only",
                    f"{return_ctx['refund_only_n']:,}",
                    f"{return_ctx['refund_only_pct']:.2f}%",
                ),
            ]
        )

        sub = st.tabs(
            [
                "Return Reasons",
                "Top10 退货款式",
                "Top5 产品链接",
                "指定链接占比",
                "Collection 链接",
                "退货金额",
                "直播 vs 非直播对比",
            ]
        )

        with sub[0]:
            st.dataframe(
                style_pct_df(return_ctx["reason_df"].head(top_n)),
                use_container_width=True,
                hide_index=True,
            )

        with sub[1]:
            st.dataframe(
                style_pct_df(return_ctx["sku_top10"]),
                use_container_width=True,
                hide_index=True,
            )

        with sub[2]:
            st.dataframe(
                style_pct_df(return_ctx["product_top5"]),
                use_container_width=True,
                hide_index=True,
            )

        with sub[3]:
            st.dataframe(
                style_pct_df(return_ctx["target_link_df"]),
                use_container_width=True,
                hide_index=True,
            )

        with sub[4]:
            st.dataframe(
                style_pct_df(return_ctx["collection_df"]),
                use_container_width=True,
                hide_index=True,
            )

        with sub[5]:
            tra = return_ctx.get("total_return_amount", np.nan)
            ara = return_ctx.get("avg_return_amount", np.nan)
            c1r, c2r = st.columns(2)
            c1r.metric("总退货金额", f"${tra:,.2f}" if pd.notna(tra) and tra > 0 else "—（未找到金额列）")
            c2r.metric("平均每单退货金额", f"${ara:,.2f}" if pd.notna(ara) and ara > 0 else "—（未找到金额列）")
            st.caption("优先读 Refund Total / Return Amount 列；若无则用订单总表 Order Amount 匹配。")

        with sub[6]:
            ret_lvn = return_ctx.get("ret_live_vs_nonlive", {})
            if ret_lvn:
                live_r = ret_lvn.get("live", {})
                nonlive_r = ret_lvn.get("nonlive", {})
                unk_r = ret_lvn.get("unknown", {})
                ret_comp = [
                    ["退货包裹数", live_r.get("n",0), nonlive_r.get("n",0), unk_r.get("n",0)],
                    ["Seller Fault 数", live_r.get("seller_fault",0), nonlive_r.get("seller_fault",0), unk_r.get("seller_fault",0)],
                    ["首要退货原因", live_r.get("top_reason","—"), nonlive_r.get("top_reason","—"), unk_r.get("top_reason","—")],
                    ["平均退货 AOV", f"${live_r.get('aov',0):,.2f}" if live_r.get('aov') else "—", f"${nonlive_r.get('aov',0):,.2f}" if nonlive_r.get('aov') else "—", "—"],
                    ["1件退货订单", live_r.get("q1",0), nonlive_r.get("q1",0), unk_r.get("q1",0)],
                    ["2件退货订单", live_r.get("q2",0), nonlive_r.get("q2",0), unk_r.get("q2",0)],
                    ["3件退货订单", live_r.get("q3",0), nonlive_r.get("q3",0), unk_r.get("q3",0)],
                    ["4件+退货订单", live_r.get("q4p",0), nonlive_r.get("q4p",0), unk_r.get("q4p",0)],
                ]
                st.dataframe(pd.DataFrame(ret_comp, columns=["指标","🔴 直播时段","🔵 非直播时段","❓ Unknown"]), use_container_width=True, hide_index=True)
            else:
                st.info("需要 Returned 表的 Order ID 能匹配到订单总表 Created Time 才能归因。")

    idx += 1


if auction_ctx:
    with tabs[idx]:
        st.header("Auction 订单分析")

        show_metric_row(
            [
                ("总 Auction Order 数", f"{auction_ctx['total']:,}", None),
                (
                    "Cancelled 订单",
                    f"{auction_ctx['cancelled_n']:,}",
                    f"{pct(auction_ctx['cancelled_n'], auction_ctx['total']):.2f}%",
                ),
                (
                    "申请退货（Return/Refund）",
                    f"{auction_ctx['return_n']:,}",
                    f"退货率 {auction_ctx['return_rate']:.2f}%",
                ),
                (
                    "有效 Auction 平均 AOV",
                    fmt_money(auction_ctx["valid_aov"]),
                    "排除 Cancelled",
                ),
            ]
        )

        show_metric_row(
            [
                (
                    "提交退货 Auction 平均 AOV",
                    fmt_money(auction_ctx["return_aov"]),
                    None,
                ),
                (
                    "退货 AOV - 有效 AOV",
                    fmt_money(auction_ctx["aov_diff"]),
                    None,
                ),
            ]
        )

        st.table(
            pd.DataFrame(
                [
                    ["总 Order 数", auction_ctx["total"]],
                    ["Cancelled 订单", auction_ctx["cancelled_n"]],
                    ["申请退货（Return/Refund）", auction_ctx["return_n"]],
                    [
                        "退货率",
                        f"{auction_ctx['return_n']} / {auction_ctx['total']} = {auction_ctx['return_rate']:.2f}%",
                    ],
                    [
                        "有效 Auction 平均 AOV（排除 Cancelled）",
                        fmt_money(auction_ctx["valid_aov"]),
                    ],
                    [
                        "提交退货 Auction 平均 AOV",
                        fmt_money(auction_ctx["return_aov"]),
                    ],
                ],
                columns=["指标", "数值"],
            )
        )

        sub = st.tabs(
            [
                "Auction SKU 分布",
                "Cancel Reasons",
                "Cancel SKU",
                "Cancel 产品链接",
                "Return Reasons",
                "Return SKU",
                "Return 产品链接",
                "直播 vs 非直播对比",
            ]
        )

        with sub[0]:
            st.dataframe(
                style_pct_df(auction_ctx["sku_dist"]),
                use_container_width=True,
                hide_index=True,
            )

        with sub[1]:
            st.dataframe(
                style_pct_df(auction_ctx["cancel_reason"]),
                use_container_width=True,
                hide_index=True,
            )

        with sub[2]:
            st.dataframe(
                style_pct_df(auction_ctx.get("cancel_sku_dist", pd.DataFrame())),
                use_container_width=True,
                hide_index=True,
            )

        with sub[3]:
            st.dataframe(
                style_pct_df(auction_ctx.get("cancel_product", pd.DataFrame())),
                use_container_width=True,
                hide_index=True,
            )

        with sub[4]:
            st.dataframe(
                style_pct_df(auction_ctx["return_reason"]),
                use_container_width=True,
                hide_index=True,
            )

        with sub[5]:
            st.dataframe(
                style_pct_df(auction_ctx.get("return_sku", pd.DataFrame())),
                use_container_width=True,
                hide_index=True,
            )

        with sub[6]:
            st.dataframe(
                style_pct_df(auction_ctx.get("return_product", pd.DataFrame())),
                use_container_width=True,
                hide_index=True,
            )

        with sub[7]:
            auc_lvn = auction_ctx.get("live_vs_nonlive", {})
            if auc_lvn:
                live_a = auc_lvn.get("live", {})
                nonlive_a = auc_lvn.get("nonlive", {})
                auc_comp = [
                    ["订单量", live_a.get("n",0), nonlive_a.get("n",0)],
                    ["取消单数", live_a.get("cancel_n",0), nonlive_a.get("cancel_n",0)],
                    ["取消率", f"{live_a.get('cancel_rate',0):.2f}%", f"{nonlive_a.get('cancel_rate',0):.2f}%"],
                    ["有效 AOV", f"${live_a.get('aov',0):,.2f}" if live_a.get('aov') else "—", f"${nonlive_a.get('aov',0):,.2f}" if nonlive_a.get('aov') else "—"],
                    ["1套占比", f"{live_a.get('q1',0)/max(live_a.get('n',1),1)*100:.1f}%", f"{nonlive_a.get('q1',0)/max(nonlive_a.get('n',1),1)*100:.1f}%"],
                    ["2套占比", f"{live_a.get('q2',0)/max(live_a.get('n',1),1)*100:.1f}%", f"{nonlive_a.get('q2',0)/max(nonlive_a.get('n',1),1)*100:.1f}%"],
                    ["3套占比", f"{live_a.get('q3',0)/max(live_a.get('n',1),1)*100:.1f}%", f"{nonlive_a.get('q3',0)/max(nonlive_a.get('n',1),1)*100:.1f}%"],
                    ["4套+占比", f"{live_a.get('q4p',0)/max(live_a.get('n',1),1)*100:.1f}%", f"{nonlive_a.get('q4p',0)/max(nonlive_a.get('n',1),1)*100:.1f}%"],
                    ["1套 AOV", f"${live_a.get('aov_q1',0):,.2f}" if live_a.get('aov_q1') else "—", f"${nonlive_a.get('aov_q1',0):,.2f}" if nonlive_a.get('aov_q1') else "—"],
                    ["2套 AOV", f"${live_a.get('aov_q2',0):,.2f}" if live_a.get('aov_q2') else "—", f"${nonlive_a.get('aov_q2',0):,.2f}" if nonlive_a.get('aov_q2') else "—"],
                    ["3套 AOV", f"${live_a.get('aov_q3',0):,.2f}" if live_a.get('aov_q3') else "—", f"${nonlive_a.get('aov_q3',0):,.2f}" if nonlive_a.get('aov_q3') else "—"],
                    ["4套+ AOV", f"${live_a.get('aov_q4p',0):,.2f}" if live_a.get('aov_q4p') else "—", f"${nonlive_a.get('aov_q4p',0):,.2f}" if nonlive_a.get('aov_q4p') else "—"],
                ]
                st.dataframe(pd.DataFrame(auc_comp, columns=["指标","🔴 直播时段","🔵 非直播时段"]), use_container_width=True, hide_index=True)
            else:
                st.info("需要 Auction 订单表含 Created Time 和 Order Amount 列才能计算。")

    idx += 1


with tabs[idx]:
    st.header("Collection 链接综合分析")

    st.markdown("#### 明细表")
    st.dataframe(
        style_pct_df(comp_df),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### 汇总表（按渠道类型）")
    st.dataframe(
        style_pct_df(channel_summary),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### 启示")

    for note in insights:
        st.markdown(note)

    action_df = pd.DataFrame(
        [
            [
                "达人带货",
                "收货后退货率高 → 内容/实物预期差",
                "审查达人展示内容，强化真实买家评价和细节展示",
            ],
            [
                "官号视频",
                "Cancel 率高 → 下单后反悔 / 等待耐心差",
                "优化 Buy 4 促销设计、发货时效和视频承诺",
            ],
            [
                "直播间",
                "冲动购买结构性问题",
                "加强尺码选择、材质展示和售前提醒，减少 No Longer Needed",
            ],
        ],
        columns=["渠道", "核心问题", "建议方向"],
    )

    st.markdown("#### 渠道动作建议表")
    st.dataframe(
        action_df,
        use_container_width=True,
        hide_index=True,
    )

idx += 1


with tabs[idx]:
    st.header("Downloads")

    sheets = [
        ("Cancel Reasons", cancel_ctx.get("reason_df")),
        ("Cancel SKU", cancel_ctx.get("sku_breakdown")),
        ("Cancel Product Links", cancel_ctx.get("product_breakdown")),
        ("Cancel Collection Links", cancel_ctx.get("collection_df")),
        ("Cancel Live", cancel_ctx.get("live_summary")),
        ("Collection Comparison", comp_df),
        ("Collection Channel Summary", channel_summary),
        ("Channel Action", action_df),
    ]

    if return_ctx:
        sheets.extend(
            [
                ("Return Reasons", return_ctx.get("reason_df")),
                ("Return SKU Top10", return_ctx.get("sku_top10")),
                ("Return Product Top5", return_ctx.get("product_top5")),
                ("Return Target Links", return_ctx.get("target_link_df")),
                ("Return Collection Links", return_ctx.get("collection_df")),
            ]
        )

    if auction_ctx:
        sheets.extend(
            [
                ("Auction SKU", auction_ctx.get("sku_dist")),
                ("Auction Cancel Reasons", auction_ctx.get("cancel_reason")),
                ("Auction Cancel SKU", auction_ctx.get("cancel_sku_dist")),
                ("Auction Cancel Products", auction_ctx.get("cancel_product")),
                ("Auction Return Reason", auction_ctx.get("return_reason")),
                ("Auction Return SKU", auction_ctx.get("return_sku")),
                ("Auction Return Products", auction_ctx.get("return_product")),
            ]
        )

    # Build individual HTML reports
    cancelled_html = build_cancelled_html(cancel_ctx, top_n)
    returned_html = build_returned_html(return_ctx, top_n) if return_ctx else None
    auction_html = build_auction_html(auction_ctx, top_n) if auction_ctx else None
    collection_html = build_collection_html(comp_df, channel_summary, insights, action_df)
    combined_html = build_html_report(
        cancel_ctx=cancel_ctx,
        return_ctx=return_ctx,
        auction_ctx=auction_ctx,
        comp_df=comp_df,
        channel_summary=channel_summary,
        insights=insights,
        action_df=action_df,
        top_n=top_n,
    )

    st.subheader("📥 Excel")
    st.download_button(
        "下载 Excel Report（全部 Sheet）",
        data=excel_bytes(sheets),
        file_name="cancel_return_auction_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("📄 Individual HTML Reports")
    st.caption("每个模块单独一份 HTML，内容与示例报告对齐（含完整分布表、核心指标板块）。")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📊 Cancelled Orders Report HTML",
            data=cancelled_html.encode("utf-8"),
            file_name="cancelled_report.html",
            mime="text/html",
        )
    with col2:
        if returned_html:
            st.download_button(
                "📦 Returned Orders Report HTML",
                data=returned_html.encode("utf-8"),
                file_name="returned_report.html",
                mime="text/html",
            )
        else:
            st.info("上传 Returned Orders 表后解锁")

    col3, col4 = st.columns(2)
    with col3:
        if auction_html:
            st.download_button(
                "🔨 Auction Orders Report HTML",
                data=auction_html.encode("utf-8"),
                file_name="auction_report.html",
                mime="text/html",
            )
        else:
            st.info("上传 Auction Orders 表后解锁")
    with col4:
        st.download_button(
            "📈 Collection 综合分析 HTML",
            data=collection_html.encode("utf-8"),
            file_name="collection_report.html",
            mime="text/html",
        )

    st.subheader("📄 Combined HTML Report（全部合并）")
    st.download_button(
        "下载合并版 HTML Report",
        data=combined_html.encode("utf-8"),
        file_name="cancel_return_auction_report.html",
        mime="text/html",
    )

    with st.expander("预览合并版 HTML Report", expanded=False):
        components.html(combined_html, height=900, scrolling=True)
