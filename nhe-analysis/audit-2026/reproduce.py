#!/usr/bin/env python3
"""Independent CMS NHE24 summary audit; never modifies original analysis."""
from pathlib import Path
import csv, json, math, statistics

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/"data"/"NHE24_Summary.csv"
OUT=ROOT/"results"

def corr(a,b):
    am,bm=statistics.mean(a),statistics.mean(b)
    return sum((x-am)*(y-bm) for x,y in zip(a,b))/math.sqrt(
        sum((x-am)**2 for x in a)*sum((y-bm)**2 for y in b))

def main():
    with SOURCE.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.reader(f))
    headers=[r for r in rows if len(r)>4 and r[1:4]==["1960","1961","1962"]]
    assert len(headers)==1,"Cannot identify unique year header"
    years=[int(v) for v in headers[0][1:] if v.isdigit()]
    assert years==list(range(1960,2025)),"Unexpected source years"
    def get(label):
        hits=[r for r in rows if r and r[0].strip()==label]
        assert len(hits)==1,(label,len(hits))
        values=hits[0][1:len(years)+1]
        assert len(values)==len(years)
        return {y:float(v.replace(",","")) for y,v in zip(years,values)}
    n=get("National Health Expenditures (Amount in Billions)")
    a=get("Government Administration and Non-Medical Insurance Expenditures")
    years80=[y for y in years if y>=1980]
    result={
      "source":"Archived NHE24_Summary.csv; nominal USD billions",
      "nhe_2024_billion":n[2024],
      "admin_2024_billion":a[2024],
      "admin_1990_billion":a[1990],
      "admin_share_2024_pct":100*a[2024]/n[2024],
      "admin_nominal_growth_1990_2024_pct":100*(a[2024]/a[1990]-1),
      "admin_share_by_year_pct":{str(y):100*a[y]/n[y] for y in [1990,2000,2010,2020,2024]},
      "pearson_nominal_1960_2024":corr([n[y] for y in years],[a[y] for y in years]),
      "pearson_nominal_1980_2024":corr([n[y] for y in years80],[a[y] for y in years80]),
      "pearson_annual_log_growth":corr(
          [math.log(n[y]/n[y-1]) for y in years[1:]],
          [math.log(a[y]/a[y-1]) for y in years[1:]]),
      "scenario_2030_billion_5_8pct":n[2024]*1.058**6,
      "scenario_2033_billion_5_8pct":n[2024]*1.058**9,
      "limitations":["Nominal series; no inflation adjustment",
                     "Correlation is not causal evidence",
                     "CMS category is not a measure of waste",
                     "5.8% growth is illustrative, not official CMS forecast"]
    }
    OUT.mkdir(exist_ok=True)
    (OUT/"summary_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    with (OUT/"illustrative_scenario.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["year","nhe_billion","classification"])
        for y in range(2024,2034):
            w.writerow([y,round(n[2024]*1.058**(y-2024),3),
                        "actual baseline" if y==2024 else "illustrative scenario"])
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
