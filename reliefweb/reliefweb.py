import pandas as pd
import requests
import re
import time
import json
from typing import List, Dict, Tuple


INPUT_FILE = "..\\data\\processed\\ibtracs_era5_20251218_1520.csv"
OUTPUT_FILE = "..\\data\\processed\\ibtracs_era5_20251218_1520_reliefweb.csv"
API_URL = "https://api.reliefweb.int/v1/reports"


APP_NAME = "Sacha-disasterdataset-94"  

# Debug to see what's happening during the API Calls
DEBUG = True
SHOW_API_RESPONSE = False
MAX_STORMS_TO_PROCESS = None  

#regex
KEYWORDS_CASUALTIES = r"\b(dead|death|killed|fatalities|lives lost|casualt|toll|deceased|perished)\b"
KEYWORDS_INJURED = r"\b(injured|wounded|hurt|treating|treated|hospitalised|hospitalized)\b"
KEYWORDS_DISPLACED = r"\b(evacuated|displaced|homeless|shelter|relocated|fleeing|forced to move|evacuation)\b"
KEYWORDS_AFFECTED = r"\b(affected|impacted|in need|vulnerable|exposed|suffering)\b"
KEYWORDS_DISEASE = r"\b(disease|cholera|malaria|dengue|cases|infections|ill|sick|outbreak|health concern|epidemic)\b"
STORM_KEYWORDS = r"\b(cyclone|hurricane|typhoon|tropical storm|tropical depression|storm|TC)\b"


def calculate_relevance_score(report: Dict, storm_name: str, year: int) -> Tuple[float, str]:
    """
    output : score to determine the relevance of each report
    """
    fields = report.get('fields', {})
    title = fields.get('title', '').lower()
    body = fields.get('body', '')
    clean_body = re.sub(r'<[^>]+>', '', body).lower()
    
    score = 0.0
    reasons = []
    
    storm_name_lower = storm_name.lower()
    year_str = str(int(year))
    if storm_name_lower in title:
        score += 0.5
        reasons.append("nom dans titre")
    else:
        return (0.1, "nom absent du titre")
    
    if re.search(STORM_KEYWORDS, title, re.IGNORECASE):
        score += 0.3
        reasons.append("mot-clé tempête dans titre")
    elif re.search(STORM_KEYWORDS, clean_body[:500], re.IGNORECASE):
        score += 0.1
        reasons.append("mot-clé tempête dans corps")
    
    title_and_start = title + " " + clean_body[:300]
    if year_str in title_and_start:
        score += 0.2
        reasons.append("année présente")
    
    storm_count = clean_body.count(storm_name_lower)
    if storm_count >= 3:
        score += 0.1
        reasons.append(f"nom mentionné {storm_count}x")
    elif storm_count >= 1:
        score += 0.05
    
    generic_patterns = [
        r"annual report",
        r"yearly report",
        r"progress report \d{4}",
        r"humanitarian response plan",
        r"flash appeal",
        r"strategic response plan"
    ]
    
    is_generic = any(re.search(pattern, title, re.IGNORECASE) for pattern in generic_patterns)
    if is_generic and storm_count < 5:
        score *= 0.3  
        reasons.append("rapport générique détecté")
    
    reason_str = ", ".join(reasons) if reasons else "aucun critère"
    return (score, reason_str)


def get_reliefweb_data(storm_name: str, year: int, limit: int = 30) -> List[Dict]:
    """
    Search with strict relevance validation
    """
    all_results = []
    
   # Search strategies (from most specific to broadest)
    strategies = [
        (f"cyclone {storm_name} {int(year)}", "Cyclone + Nom + Année"),
        (f"hurricane {storm_name} {int(year)}", "Hurricane + Nom + Année"),
        (f"typhoon {storm_name} {int(year)}", "Typhoon + Nom + Année"),
        (f"{storm_name} {int(year)}", "Nom + Année"),
    ]
    
    for query_str, description in strategies:
        if DEBUG:
            print(f"\n   🔍 Tentative: {description}")
            print(f"      Requête: '{query_str}'")
        
        url_params={
            "appname": APP_NAME
        }
        payload = { 
            "query": {
                "value": query_str,
                "operator": "AND"
            },
            "filter": {
                "conditions": [
                    {
                        "field": "date.created",
                        "value": {
                            "from": f"{int(year)}-01-01T00:00:00+00:00",
                            "to": f"{int(year)}-12-31T23:59:59+00:00"
                        }
                    }
                ]
            },
            "fields": {
                "include": ["title", "body", "date", "url", "primary_country", "format"]
            },
            "limit": limit,
            "sort": ["date:desc"]
        }
        
        if SHOW_API_RESPONSE and DEBUG:
            print(f"      📤 Payload: {json.dumps(payload, indent=2)[:300]}...")

        try:
            response = requests.post(API_URL, params=url_params, json=payload, timeout=15)
            
            if DEBUG:
                print(f"      📥 Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json().get('data', [])
                
                if DEBUG:
                    print(f"       Raw results: {len(data)}")
                
                if data:
                    # FILTER BY RELEVANCE
                    relevant_reports = []
                    for report in data:
                        score, reason = calculate_relevance_score(report, storm_name, year)
                        
                        if DEBUG and score > 0:
                            title = report['fields'].get('title', 'N/A')
                            print(f"         • Score {score:.2f}: {title[:60]}...")
                            print(f"           Raison: {reason}")
                        
                        if score >= 0.5:
                            relevant_reports.append(report)
                    
                    if relevant_reports:
                        if DEBUG:
                            print(f"      ✅ {len(relevant_reports)} relevant reports kept (out of {len(data)})")
                        all_results.extend(relevant_reports)
                        break  
                    else:
                        if DEBUG:
                            print(f"      ❌ No relevant reports (all filtered out)")
                else:
                    if DEBUG:
                        print(f"      ❌ No results")
            else:
                if DEBUG:
                    print(f"      ❌ HTTP error: {response.status_code}")
                    print(f"         {response.text[:200]}")
            
            time.sleep(0.3)
            
        except Exception as e:
            print(f"      ⚠️ Exception: {e}")
            continue
    
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r['fields'].get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)
    
    if DEBUG and unique_results:
        print(f"   ✅ FINAL RESULT: {len(unique_results)} relevant and unique reports")
    
    return unique_results[:limit]


def extract_numbers_from_sentence(sentence: str) -> List[int]:
    """
    Extract all numbers from sentences
    """
    numbers = re.findall(r'\b\d{1,3}(?:[,\s]\d{3})*\b', sentence)
    cleaned = []
    for n in numbers:
        try:
            cleaned.append(int(n.replace(',', '').replace(' ', '')))
        except:
            pass
    return cleaned



def extract_impact_data(reports: List[Dict], storm_name: str) -> Dict[str, str]:
    """
    Extract only the numeric values (max) for each impact category.
    """
    extracted_numbers = {
        "Deaths": [],
        "Injured": [],
        "Displaced_Evacuated": [],
        "Affected": [],
        "Disease_Context": [], 
        "URL": []
    }

    storm_name_lower = storm_name.lower()

    for report in reports:
        body = report['fields'].get('body', '')
        title = report['fields'].get('title', '')
        url = report['fields'].get('url', '')
        
        full_text = f"{title}. {body}"
        clean_text = re.sub(r'<[^>]+>', '', full_text).replace('\n', ' ')
        paragraphs = clean_text.split('.')
        
        for para in paragraphs:
            para_lower = para.lower()
            is_relevant_paragraph = (
                storm_name_lower in para_lower or 
                re.search(STORM_KEYWORDS, para_lower)
            )
            if not is_relevant_paragraph and len(para) > 100:
                continue
            
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if len(sentence) < 20:
                    continue
                has_number = re.search(r'\d+', sentence)
                s_lower = sentence.lower()
                
                # Dead, Injured, Displaced, Affected
                if has_number:
                    if re.search(KEYWORDS_CASUALTIES, s_lower):
                        nums = extract_numbers_from_sentence(sentence)
                        if nums:
                            extracted_numbers["Deaths"].extend(nums)
                    if re.search(KEYWORDS_INJURED, s_lower):
                        nums = extract_numbers_from_sentence(sentence)
                        if nums:
                            extracted_numbers["Injured"].extend(nums)
                    if re.search(KEYWORDS_DISPLACED, s_lower):
                        nums = extract_numbers_from_sentence(sentence)
                        if nums:
                            extracted_numbers["Displaced_Evacuated"].extend(nums)
                    if re.search(KEYWORDS_AFFECTED, s_lower):
                        nums = extract_numbers_from_sentence(sentence)
                        if nums:
                            extracted_numbers["Affected"].extend(nums)
                
                if re.search(KEYWORDS_DISEASE, s_lower):
                    extracted_numbers["Disease_Context"].append(sentence.strip())
        
        if url and url not in extracted_numbers["URL"]:
            extracted_numbers["URL"].append(url)

    def max_or_na(lst, threshold=0):
        return str(max(lst)) if lst else "N/A"

    return {
        "RW_Casualty_Info": max_or_na(extracted_numbers["Deaths"]),
        "RW_Injured_Info": max_or_na(extracted_numbers["Injured"]),
        "RW_Evacuated_Displaced": max_or_na(extracted_numbers["Displaced_Evacuated"]),
        "RW_Affected_Population": max_or_na(extracted_numbers["Affected"]),
        "RW_Disease_Context": " | ".join(extracted_numbers["Disease_Context"][:2]) if extracted_numbers["Disease_Context"] else "N/A",
        "RW_Report_Link": extracted_numbers["URL"][0] if extracted_numbers["URL"] else "N/A",
        "RW_Total_Reports": len(extracted_numbers["URL"])
    }



def main():
    print("=" * 80)
    print("DATASET ENRICHMENT - STRICT FILTERING VERSION")
    print("=" * 80)
    
    print("\n📂 Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"   → {len(df)} rows loaded")
    
    unique_storms = df[['Storm_Name', 'Year']].drop_duplicates()
    print(f"\n🌀 Total unique storms: {len(unique_storms)}")
    
    if MAX_STORMS_TO_PROCESS:
        unique_storms = unique_storms.head(MAX_STORMS_TO_PROCESS)
        print(f"\n⚠️ TEST MODE: Processing limited to {MAX_STORMS_TO_PROCESS} storms")
    
    reliefweb_results = {} 
    success_count = 0
    fail_count = 0
    filtered_count = 0  # Count of filtered reports

    print("\n" + "=" * 80)
    print("STARTING API REQUESTS (with relevance filtering)")
    print("=" * 80)

    for index, row in unique_storms.iterrows():
        name = str(row['Storm_Name']).strip()
        year = row['Year']
        
        if name.upper() == "UNNAMED" or name == "nan" or pd.isna(name) or name == "":
            continue

        key = (name, year)
        print(f"\n{'='*80}")
        print(f"🔍 [{index+1}/{len(unique_storms)}] {name} ({int(year)})")
        print(f"{'='*80}")
        
        reports = get_reliefweb_data(name, year)
        
        if reports:
            print(f"\n   📊 Extracting impact data...")
            extracted_info = extract_impact_data(reports, name)
            reliefweb_results[key] = extracted_info
            success_count += 1
            
            print(f"\n   ✅ SUMMARY:")
            print(f"      - Dead: {'✓' if extracted_info['RW_Casualty_Info'] != 'N/A' else '✗'}")
            print(f"      - Injured: {'✓' if extracted_info['RW_Injured_Info'] != 'N/A' else '✗'}")
            print(f"      - Displaced: {'✓' if extracted_info['RW_Evacuated_Displaced'] != 'N/A' else '✗'}")
            print(f"      - Affected: {'✓' if extracted_info['RW_Affected_Population'] != 'N/A' else '✗'}")
            print(f"      - URL: {extracted_info['RW_Report_Link'][:60]}...")
        else:
            fail_count += 1
            print(f"\n   ❌ No relevant reports found")
        
        time.sleep(0.5)

    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"✅ Storms with data: {success_count}")
    print(f"❌ Storms without data: {fail_count}")
    if success_count + fail_count > 0:
        print(f"📊 Success rate: {success_count/(success_count+fail_count)*100:.1f}%")

    print("\n🔄 Merging data...")
    def apply_rw_data(row):
        name = str(row['Storm_Name']).strip()
        year = row['Year']
        key = (name, year)
        
        default_data = {
            "RW_Casualty_Info": "N/A",
            "RW_Injured_Info": "N/A",
            "RW_Evacuated_Displaced": "N/A",
            "RW_Affected_Population": "N/A",
            "RW_Disease_Context": "N/A",
            "RW_Report_Link": "N/A",
            "RW_Total_Reports": 0
        }
        
        return pd.Series(reliefweb_results.get(key, default_data))

    rw_columns = df.apply(apply_rw_data, axis=1)
    df_enriched = pd.concat([df, rw_columns], axis=1)
    df_enriched.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "=" * 80)
    print("✅ TERMINÉ !")
    print("=" * 80)
    print(f"📁 File: {OUTPUT_FILE}")
    print(f"📊 Total rows: {len(df_enriched)}")
    print(f"\n📈 Enriched data:")
    print(f"   - Dead info: {(df_enriched['RW_Casualty_Info'] != 'N/A').sum()} rows")
    print(f"   - Injured info: {(df_enriched['RW_Injured_Info'] != 'N/A').sum()} rows")
    print(f"   - Displaced info: {(df_enriched['RW_Evacuated_Displaced'] != 'N/A').sum()} rows")

if __name__ == "__main__":
    main()