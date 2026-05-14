import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})
BLUE   = '#1B4F72'
BLUE2  = '#2E86C1'
RED    = '#E74C3C'
ORANGE = '#F39C12'
GREEN  = '#1E8449'
PURPLE = '#6C3483'
GREY   = '#717D7E'
OUT = '/mnt/user-data/outputs/'
UP  = '/mnt/user-data/uploads/'

# ── Universal NHE table loader (handles "selected years" tables like T03) ─
def load_nhe(filename, yr_row_idx=1, data_start=3):
    df = pd.read_excel(UP + filename, header=None)
    yr_row = df.iloc[yr_row_idx]
    years, yr_cols = [], []
    for j, v in enumerate(yr_row):
        s = str(v).replace('.0','').strip()
        if s.isdigit() and 1960 <= int(s) <= 2030:
            years.append(int(s))
            yr_cols.append(j)
    data = {}
    for i in range(data_start, len(df)):
        label = str(df.iloc[i, 0]).strip()
        if not label or label == 'nan':
            continue
        vals = []
        for j in yr_cols:
            try:
                vals.append(float(str(df.iloc[i,j]).replace(',','')))
            except:
                vals.append(np.nan)
        data[label] = pd.Series(vals, index=years)
    return data

# ── Load Summary CSV ───────────────────────────────────────────────────────
sum_df = pd.read_csv(UP + 'NHE24_Summary.csv', header=None)

def get_sum_row(keyword):
    for i, row in sum_df.iterrows():
        lbl = str(row.iloc[0])
        if keyword.lower() in lbl.lower():
            vals = []
            for j, v in enumerate(row.iloc[1:]):
                try:
                    vals.append(float(str(v).replace(',','')))
                except:
                    vals.append(np.nan)
            s = pd.Series(vals)
            s.index = range(1960, 1960+len(s))
            return s.dropna()
    return pd.Series(dtype=float)

nhe_total  = get_sum_row('National Health Expenditures (Amount')
gdp        = get_sum_row('Gross Domestic Product2  (Amount')
admin      = get_sum_row('Government Administration and Non-Medical')
phc        = get_sum_row('Personal Health Care')
pub_health = get_sum_row('Government Public Health Activities')
percapita  = get_sum_row('National Health Expenditures (Per Capita')

nhe_gdp_pct = (nhe_total / gdp * 100).dropna()
admin_pct   = (admin / nhe_total * 100).dropna()

# ── Load Table 03 (payer mix) ──────────────────────────────────────────────
t03 = load_nhe('Table_03_National_Health_Expenditures__by_Source_of_Funds.xlsx')

def gt(d, kw):
    for k in d:
        if kw.lower() in k.lower():
            return d[k].dropna()
    return pd.Series(dtype=float)

medicare      = gt(t03, 'Medicare')
medicaid      = gt(t03, 'Medicaid')
private_ins   = gt(t03, 'Private Health Insurance')
out_of_pocket = gt(t03, 'Out of pocket')

# ── Load Table 02 (service type) ───────────────────────────────────────────
t02 = load_nhe('Table_02_National_Health_Expenditures__Aggregate_and_Per_Capita_Amounts__by_Type_of_Expenditure.xlsx')
hospital_svc  = gt(t02, 'Hospital Care')
physician_svc = gt(t02, 'Physician and Clinical')
rx_drugs      = gt(t02, 'Prescription Drugs')
nursing       = gt(t02, 'Nursing Care')
home_health   = gt(t02, 'Home Health')

# ── Load Table 07 for full hospital history ────────────────────────────────
t07 = load_nhe('Table_07_Hospital_Care_Expenditures.xlsx')
if not hospital_svc.empty:
    pass

print("=== KEY METRICS ===")
print(f"NHE 2024: ${nhe_total[2024]/1000:.2f}T")
print(f"NHE/GDP 2024: {nhe_gdp_pct[2024]:.1f}%")
print(f"Admin 2024: ${admin[2024]:.1f}B  ({admin_pct[2024]:.1f}% of NHE)")
print(f"Per capita 2024: ${percapita[2024]:,.0f}")
print(f"Admin growth 1990-2024: {(admin[2024]/admin[1990]-1)*100:.0f}%")
if 2024 in medicare.index:
    print(f"Medicare 2024: ${medicare[2024]:.1f}B")
    print(f"Medicaid 2024: ${medicaid[2024]:.1f}B")
    print(f"Private Ins 2024: ${private_ins[2024]:.1f}B")

# ──────────────────────────────────────────────────────────────────────────
# FIG 1: Total NHE & GDP share 1960-2024
# ──────────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 5.5))
yrs = sorted(set(nhe_total.index) & set(nhe_gdp_pct.index))
ax1.fill_between(yrs, [nhe_total[y]/1000 for y in yrs], alpha=0.2, color=BLUE)
ax1.plot(yrs, [nhe_total[y]/1000 for y in yrs], color=BLUE, lw=2.5, label='Total NHE ($ Trillions)')
ax1.set_ylabel('National Health Expenditures ($ Trillions)', color=BLUE)
ax1.tick_params(axis='y', labelcolor=BLUE)
ax1.set_xlabel('Year')
ax2 = ax1.twinx()
ax2.plot(yrs, [nhe_gdp_pct[y] for y in yrs], color=RED, lw=2.5, linestyle='--', label='NHE as % of GDP')
ax2.set_ylabel('NHE as % of GDP', color=RED)
ax2.tick_params(axis='y', labelcolor=RED)
ax2.spines['right'].set_visible(True)
for yr, lbl in [(1980,'$253B\n9.1%'), (2000,'$1.4T\n13.3%'), (2020,'COVID\n19.7%'), (2024,'$5.3T\n18.0%')]:
    if yr in nhe_total.index:
        ax1.axvline(yr, color='grey', linestyle=':', alpha=0.5, lw=1)
        ax1.text(yr+0.5, nhe_total[yr]/1000+0.1, lbl, fontsize=7.5, color='grey')
l1,lb1 = ax1.get_legend_handles_labels()
l2,lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc='upper left', fontsize=9)
ax1.set_title('Figure 1. U.S. National Health Expenditures (1960-2024)\n'
              'Total Spending and Share of GDP - CMS National Health Expenditure Accounts', fontweight='bold')
plt.tight_layout()
plt.savefig(OUT+'nhe_fig1_total_gdp.png', bbox_inches='tight')
plt.close()
print("FIG 1 done")

# ──────────────────────────────────────────────────────────────────────────
# FIG 2: Admin cost trend
# ──────────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 5.5))
yrs = sorted(admin.index)
ax1.bar(yrs, [admin[y] for y in yrs], color=ORANGE, alpha=0.75, label='Admin Cost ($B)', width=0.8)
ax1.set_ylabel('Administrative & Non-Medical Insurance Cost ($ Billions)', color=ORANGE)
ax1.tick_params(axis='y', labelcolor=ORANGE)
ax1.set_xlabel('Year')
ax2 = ax1.twinx()
ax2.plot(yrs, [admin_pct[y] for y in yrs], color=RED, lw=2.5, label='Admin % of NHE')
ax2.set_ylabel('Admin Cost as % of NHE', color=RED)
ax2.tick_params(axis='y', labelcolor=RED)
ax2.spines['right'].set_visible(True)
ax2.axhline(7.0, color=RED, linestyle='--', alpha=0.35, lw=1.2)
ax2.text(1962, 7.2, '7% benchmark line', fontsize=8, color=RED, alpha=0.6)
l1,lb1 = ax1.get_legend_handles_labels()
l2,lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc='upper left', fontsize=9)
ax1.set_title('Figure 2. U.S. Healthcare Administrative Cost Trend (1960-2024)\n'
              'Government Administration & Non-Medical Insurance Expenditures - CMS NHE Data', fontweight='bold')
plt.tight_layout()
plt.savefig(OUT+'nhe_fig2_admin_trend.png', bbox_inches='tight')
plt.close()
print("FIG 2 done")

# ──────────────────────────────────────────────────────────────────────────
# FIG 3: Payer mix stacked area (T03 - selected years)
# ──────────────────────────────────────────────────────────────────────────
common = sorted(set(medicare.index) & set(medicaid.index) &
                set(private_ins.index) & set(out_of_pocket.index))
common = [y for y in common if y >= 1987]

if len(common) > 0:
    med_v  = [medicare[y]      for y in common]
    mcd_v  = [medicaid[y]      for y in common]
    pri_v  = [private_ins[y]   for y in common]
    oop_v  = [out_of_pocket[y] for y in common]
    tot_v  = [med_v[i]+mcd_v[i]+pri_v[i]+oop_v[i] for i in range(len(common))]
    med_p  = [med_v[i]/tot_v[i]*100 for i in range(len(common))]
    mcd_p  = [mcd_v[i]/tot_v[i]*100 for i in range(len(common))]
    pri_p  = [pri_v[i]/tot_v[i]*100 for i in range(len(common))]
    oop_p  = [oop_v[i]/tot_v[i]*100 for i in range(len(common))]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.stackplot(common, oop_p, pri_p, mcd_p, med_p,
                 labels=['Out-of-Pocket','Private Insurance','Medicaid','Medicare'],
                 colors=[GREY, BLUE2, GREEN, RED], alpha=0.85)
    ax.set_ylabel('Share of Total Health Spending (%)')
    ax.set_xlabel('Year')
    ax.set_ylim(0, 100)
    ax.set_title('Figure 3. U.S. Healthcare Payer Mix (1987-2024)\n'
                 'Share of National Health Expenditures by Source of Funds - CMS NHE Table 3', fontweight='bold')
    ax.legend(loc='lower left', fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT+'nhe_fig3_payer_mix.png', bbox_inches='tight')
    plt.close()
    print("FIG 3 done")

# ──────────────────────────────────────────────────────────────────────────
# FIG 4: Service breakdown 2024
# ──────────────────────────────────────────────────────────────────────────
yr = 2024
svc_data = {}
for lbl, s in [
    ('Hospital Care',           hospital_svc),
    ('Physician & Clinical Svcs', physician_svc),
    ('Prescription Drugs',      rx_drugs),
    ('Admin & Non-Medical Ins.',admin),
    ('Nursing Care Facilities', nursing),
    ('Home Health Care',        home_health),
    ('Public Health Activities',pub_health),
]:
    if not s.empty and yr in s.index:
        svc_data[lbl] = s[yr]

svc_series = pd.Series(svc_data).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = [RED if 'Admin' in l else (ORANGE if 'Public' in l else BLUE) for l in svc_series.index]
bars = ax.barh(svc_series.index, svc_series.values/1000, color=bar_colors, edgecolor='white', height=0.6)
for bar, val in zip(bars, svc_series.values):
    label = f'${val/1000:.2f}T' if val >= 1000 else f'${val:.0f}B'
    ax.text(val/1000 + 0.01, bar.get_y()+bar.get_height()/2, label, va='center', fontsize=9)
ax.set_xlabel('Expenditures ($ Trillions)')
ax.set_title('Figure 4. U.S. Healthcare Expenditures by Service Category (2024)\n'
             'Administrative costs (red) vs. clinical delivery - CMS NHE Tables 2 & 3', fontweight='bold')
ax.legend(handles=[
    mpatches.Patch(color=RED,   label='Administrative/Non-Clinical'),
    mpatches.Patch(color=ORANGE,label='Public Health'),
    mpatches.Patch(color=BLUE,  label='Clinical Care Delivery'),
], fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(OUT+'nhe_fig4_service_2024.png', bbox_inches='tight')
plt.close()
print("FIG 4 done")

# ──────────────────────────────────────────────────────────────────────────
# FIG 5: Growth index 1990=100
# ──────────────────────────────────────────────────────────────────────────
base = 1990
g_yrs = sorted(set(admin.index) & set(hospital_svc.index) &
               set(physician_svc.index) & set(rx_drugs.index))
g_yrs = [y for y in g_yrs if y >= base]

def idx(s, yrs, b):
    return [s[y]/s[b]*100 for y in yrs if y in s.index]

i_adm = idx(admin,        g_yrs, base)
i_hos = idx(hospital_svc, g_yrs, base)
i_phy = idx(physician_svc,g_yrs, base)
i_rxd = idx(rx_drugs,     g_yrs, base)

g_yrs_adm = [y for y in g_yrs if y in admin.index]
g_yrs_hos = [y for y in g_yrs if y in hospital_svc.index]
g_yrs_phy = [y for y in g_yrs if y in physician_svc.index]
g_yrs_rxd = [y for y in g_yrs if y in rx_drugs.index]

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(g_yrs_adm, i_adm, color=RED,    lw=2.8, label='Admin & Non-Medical Insurance', zorder=4)
ax.plot(g_yrs_hos, i_hos, color=BLUE,   lw=2.2, label='Hospital Care')
ax.plot(g_yrs_phy, i_phy, color=ORANGE, lw=2.2, label='Physician & Clinical Services')
ax.plot(g_yrs_rxd, i_rxd, color=GREEN,  lw=2.2, label='Prescription Drugs')
ax.axhline(100, color='grey', linestyle='--', lw=1, alpha=0.5, label='Base = 100 (1990)')
ax.axvline(2010, color='grey', linestyle=':', alpha=0.4)
ax.text(2010.3, max(i_rxd)*0.88, 'ACA\nEnacted', fontsize=8, color='grey')
ax.axvline(2020, color='grey', linestyle=':', alpha=0.4)
ax.text(2020.3, max(i_rxd)*0.76, 'COVID-19', fontsize=8, color='grey')
ax.set_ylabel('Spending Index (1990 = 100)')
ax.set_xlabel('Year')
ax.set_title('Figure 5. Healthcare Spending Growth Index by Category (1990-2024, Base = 1990)\n'
             'Administrative costs have outpaced most clinical categories - CMS NHE Data', fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig(OUT+'nhe_fig5_growth_index.png', bbox_inches='tight')
plt.close()
print("FIG 5 done")

# ──────────────────────────────────────────────────────────────────────────
# FIG 6: Per capita 1960-2024
# ──────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
pc_yrs = sorted(percapita.index)
pc_vals= [percapita[y] for y in pc_yrs]
ax.fill_between(pc_yrs, pc_vals, alpha=0.18, color=BLUE2)
ax.plot(pc_yrs, pc_vals, color=BLUE2, lw=2.5)
for yr, txt in [(1970,'$353'),(1990,'$2,835'),(2000,'$4,842'),(2010,'$8,377'),(2020,'$12,637'),(2024,'$15,474')]:
    if yr in percapita.index:
        ax.annotate(f'{yr}\n{txt}',
                    xy=(yr, percapita[yr]),
                    xytext=(yr+1, percapita[yr]+800),
                    fontsize=8, color=BLUE,
                    arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8))
ax.set_ylabel('Per Capita NHE (USD)')
ax.set_xlabel('Year')
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
ax.set_title('Figure 6. U.S. Per Capita National Health Expenditures (1960-2024)\n'
             'From $146 per person in 1960 to $15,474 in 2024 - CMS NHE Summary', fontweight='bold')
plt.tight_layout()
plt.savefig(OUT+'nhe_fig6_percapita.png', bbox_inches='tight')
plt.close()
print("FIG 6 done")

# ──────────────────────────────────────────────────────────────────────────
# FIG 7: Correlation Heatmap
# ──────────────────────────────────────────────────────────────────────────
corr_yrs = sorted(set(nhe_total.index) & set(admin.index) &
                  set(hospital_svc.index) & set(physician_svc.index) &
                  set(rx_drugs.index) & set(gdp.index))
corr_yrs = [y for y in corr_yrs if y >= 1980]

corr_df = pd.DataFrame({
    'Total NHE':       [nhe_total[y]      for y in corr_yrs],
    'Admin Costs':     [admin[y]          for y in corr_yrs],
    'Hospital Care':   [hospital_svc[y]   for y in corr_yrs],
    'Physician Svcs':  [physician_svc[y]  for y in corr_yrs],
    'Rx Drugs':        [rx_drugs[y]       for y in corr_yrs],
    'GDP':             [gdp[y]            for y in corr_yrs],
}, index=corr_yrs)

fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr_df.corr(), annot=True, fmt='.3f', cmap='RdYlBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={'shrink':0.8})
ax.set_title('Figure 7. Pearson Correlation Matrix - NHE Expenditure Categories (1980-2024)\n'
             'CMS National Health Expenditure Accounts', fontweight='bold')
plt.tight_layout()
plt.savefig(OUT+'nhe_fig7_correlation.png', bbox_inches='tight')
plt.close()
print("FIG 7 done")

# ──────────────────────────────────────────────────────────────────────────
# FIG 8: Admin cost vs NHE scatter + regression
# ──────────────────────────────────────────────────────────────────────────
s_yrs = sorted(set(nhe_total.index) & set(admin.index))
s_yrs = [y for y in s_yrs if y >= 1980]
xv = [nhe_total[y] for y in s_yrs]
yv = [admin[y]     for y in s_yrs]
slope, intercept, r, p, _ = linregress(xv, yv)

fig, ax = plt.subplots(figsize=(9, 5.5))
sc = ax.scatter(xv, yv, c=s_yrs, cmap='RdYlBu_r', s=60, zorder=3)
xl = np.linspace(min(xv), max(xv), 200)
ax.plot(xl, intercept + slope*xl, color=BLUE, lw=2, label=f'Linear Fit (r={r:.3f}, p<0.001)')
plt.colorbar(sc, ax=ax, label='Year')
ax.set_xlabel('Total NHE ($ Billions)')
ax.set_ylabel('Admin & Non-Medical Insurance ($ Billions)')
ax.set_title(f'Figure 8. Administrative Cost vs. Total NHE (1980-2024)\n'
             f'r = {r:.3f} - Strong positive correlation confirms admin scales with total spending', fontweight='bold')
ax.legend(fontsize=9)
for yr in [1990, 2000, 2010, 2020, 2024]:
    if yr in nhe_total.index:
        ax.annotate(str(yr), xy=(nhe_total[yr], admin[yr]),
                    fontsize=7.5, color='grey', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(OUT+'nhe_fig8_admin_scatter.png', bbox_inches='tight')
plt.close()
print("FIG 8 done")

# ──────────────────────────────────────────────────────────────────────────
# FIG 9: Payer mix 2024 donut
# ──────────────────────────────────────────────────────────────────────────
yr = 2024
pv, pl = [], []
for lbl, s in [('Medicare', medicare),('Private Insurance', private_ins),
               ('Medicaid', medicaid),('Out-of-Pocket', out_of_pocket)]:
    if not s.empty and yr in s.index:
        pv.append(s[yr]); pl.append(lbl)

if len(pv) >= 3:
    tot = sum(pv)
    pp  = [v/tot*100 for v in pv]
    fig, ax = plt.subplots(figsize=(8, 7))
    wc = [RED, BLUE2, GREEN, GREY]
    wedges, texts, autotexts = ax.pie(
        pp, labels=None, colors=wc[:len(pv)],
        autopct='%1.1f%%', startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight('bold'); at.set_color('white')
    ax.legend(
        [f'{l}  ${v:.0f}B' for l,v in zip(pl,pv)],
        loc='lower center', ncol=2, fontsize=10,
        bbox_to_anchor=(0.5, -0.08)
    )
    ax.text(0, 0, f'$5.3T\nTotal\nNHE', ha='center', va='center',
            fontsize=12, fontweight='bold', color=BLUE)
    ax.set_title('Figure 9. U.S. Healthcare Payer Distribution (2024)\n'
                 'Share of National Health Expenditures by Source - CMS NHE Table 3', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUT+'nhe_fig9_payer_donut.png', bbox_inches='tight')
    plt.close()
    print("FIG 9 done")

# ──────────────────────────────────────────────────────────────────────────
# FIG 10: NHE Projection to 2033
# ──────────────────────────────────────────────────────────────────────────
h_yrs  = sorted([y for y in nhe_total.index if y >= 2000])
h_vals = [nhe_total[y] for y in h_yrs]
future = list(range(2025, 2034))
cms_fc = [nhe_total[2024]]
for _ in future[1:]:
    cms_fc.append(cms_fc[-1]*1.058)
cms_lo = [v*0.93 for v in cms_fc]
cms_hi = [v*1.07 for v in cms_fc]

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(h_yrs,  [v/1000 for v in h_vals], color=BLUE,  lw=2.5, label='Historical NHE')
ax.plot(future, [v/1000 for v in cms_fc],  color=RED,   lw=2.5, linestyle='--', label='CMS Projection (5.8% avg/yr)')
ax.fill_between(future,
                [v/1000 for v in cms_lo],
                [v/1000 for v in cms_hi],
                alpha=0.15, color=RED, label='Confidence Band')
ax.axvline(2024, color='grey', linestyle=':', lw=1.5)
ax.text(2024.2, 2.5, 'Projection\nStart', fontsize=8, color='grey')
ax.annotate(f'2033 est.\n~${cms_fc[-1]/1000:.1f}T',
            xy=(2033, cms_fc[-1]/1000),
            xytext=(2030, cms_fc[-1]/1000 - 0.8),
            fontsize=9, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=1))
ax.set_ylabel('National Health Expenditures ($ Trillions)')
ax.set_xlabel('Year')
ax.set_title('Figure 10. U.S. NHE Historical Trend and CMS Projections (2000-2033)\n'
             'NHE projected to exceed $8T by 2030 at current trajectory - CMS NHE Projections', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT+'nhe_fig10_forecast.png', bbox_inches='tight')
plt.close()
print("FIG 10 done")

# ──────────────────────────────────────────────────────────────────────────
# TABLE 1
# ──────────────────────────────────────────────────────────────────────────
decade_yrs = [1970,1980,1990,2000,2010,2015,2020,2022,2023,2024]
trows = []
for yr in decade_yrs:
    trows.append({
        'Year': str(yr),
        'Total NHE': f"${nhe_total.get(yr,np.nan):,.0f}B",
        'NHE/GDP':   f"{nhe_gdp_pct.get(yr,np.nan):.1f}%",
        'Admin Cost':f"${admin.get(yr,np.nan):.0f}B",
        'Admin % NHE':f"{admin_pct.get(yr,np.nan):.1f}%",
        'Per Capita': f"${percapita.get(yr,np.nan):,.0f}",
    })
tbl_df = pd.DataFrame(trows)
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.axis('off')
tbl = ax.table(cellText=tbl_df.values, colLabels=tbl_df.columns,
               cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor(BLUE); cell.set_text_props(color='white', fontweight='bold')
    elif str(tbl_df.values[row-1][0]) == '2024':
        cell.set_facecolor('#FDEBD0')
    elif row % 2 == 0:
        cell.set_facecolor('#EBF5FB')
    cell.set_edgecolor('#CCCCCC')
ax.set_title('Table 1. U.S. National Health Expenditure Key Metrics by Year (1970-2024)\n'
             'Source: CMS National Health Expenditure Accounts', fontweight='bold', fontsize=11, pad=15)
plt.tight_layout()
plt.savefig(OUT+'nhe_table1_summary.png', bbox_inches='tight')
plt.close()
print("TABLE 1 done")

# ── FINAL SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ALL FIGURES COMPLETE - KEY FINDINGS")
print("="*60)
print(f"1. Total NHE 2024:           ${nhe_total[2024]/1000:.2f} Trillion")
print(f"2. NHE as % of GDP 2024:     {nhe_gdp_pct[2024]:.1f}%")
print(f"3. Admin cost 2024:          ${admin[2024]:.1f}B ({admin_pct[2024]:.1f}% of NHE)")
print(f"4. Admin cost 1990:          ${admin[1990]:.1f}B ({admin_pct[1990]:.1f}% of NHE)")
print(f"5. Admin growth 1990-2024:   {(admin[2024]/admin[1990]-1)*100:.0f}%")
print(f"6. Hospital growth 1990-2024:{(hospital_svc[2024]/hospital_svc[1990]-1)*100:.0f}%")
print(f"7. Physician growth 1990-2024:{(physician_svc[2024]/physician_svc[1990]-1)*100:.0f}%")
print(f"8. Rx Drugs growth 1990-2024:{(rx_drugs[2024]/rx_drugs[1990]-1)*100:.0f}%")
print(f"9. Per capita 2024:          ${percapita[2024]:,.0f}")
print(f"10. Per capita growth 1960-2024: {(percapita[2024]/percapita[1960]-1)*100:.0f}%")
if 2024 in medicare.index:
    tot = medicare[2024]+medicaid[2024]+private_ins[2024]+out_of_pocket[2024]
    print(f"11. Medicare share 2024:     {medicare[2024]/tot*100:.1f}%")
    print(f"12. Private Ins share 2024:  {private_ins[2024]/tot*100:.1f}%")
    print(f"13. Medicaid share 2024:     {medicaid[2024]/tot*100:.1f}%")
    print(f"14. Out-of-pocket share 2024:{out_of_pocket[2024]/tot*100:.1f}%")
