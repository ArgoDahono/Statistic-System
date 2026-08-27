import random
import pandas as pd
from datetime import datetime, timedelta

# Generate a realistic long-format CSV dataset for 300 patients
random.seed(42)

# Sample Indonesian names
first_names_m = ["Budi","Andi","Rizky","Fajar","Hendra","Agus","Teguh","Arif","Eko","Ilham","Ahmad","Bayu","Galih","Rizal","Yusuf"]
first_names_f = ["Siti","Rina","Lina","Nurul","Intan","Putri","Maya","Desi","Sri","Novi","Ayu","Fitri","Ratna","Yuni","Melati","Salsabillah","Ade Rima"]
last_names_m= ["Santoso","Pratama","Wijaya","Saputra","Hidayat","Nugroho","Kurniawan","Setiawan","Prasetyo","Maulana"]
last_names_f= ["Lestari","Sari","Kartika","Wulandari","Astuti","Putri","Maulidina","Rahma"]

def random_name(gender):
    if gender == "L":
        return random.choice(first_names_m) + " " + random.choice(last_names_m)
    else:
        return random.choice(first_names_f) + " " + random.choice(last_names_f)

rows = []

start_date = datetime(2026,1,1)

for i in range(1,301):
    no_rm = f"RM{i:03d}"
    gender = random.choice(["L","P"])
    name = random_name(gender)
    age = random.randint(18,45)
    
    tgl_masuk = start_date + timedelta(days=random.randint(0,60))
    
    # base exam date
    tgl_periksa1 = tgl_masuk + timedelta(days=random.randint(0,2))
    
    # second exam maybe different day
    tgl_periksa2 = tgl_periksa1 + timedelta(days=random.choice([0,1]))
    
    # choose type
    exam_type = random.choice(["Darah Rutin","Darah Lengkap"])
    
    # Hematology parameters
    hct = random.randint(41,50)
    albumin = round(random.uniform(3.0,3.9),1)
    
    # Add hematology rows
    params_rutin = ["WBC","RBC","PLT","HCT","HGB"]
    params_lengkap = params_rutin + ["MCV","MCH","MCHC"]
    
    params = params_rutin if exam_type=="Darah Rutin" else params_lengkap
    
    for p in params:
        val = None
        if p=="WBC": val = round(random.uniform(3.5,6.5),1)
        elif p=="RBC": val = round(random.uniform(4.0,5.5),1)
        elif p=="PLT": val = random.randint(50000,150000)
        elif p=="HCT": val = hct
        elif p=="HGB": val = round(random.uniform(11,15),1)
        elif p=="MCV": val = random.randint(80,95)
        elif p=="MCH": val = random.randint(27,32)
        elif p=="MCHC": val = random.randint(32,36)
        
        rows.append([no_rm,name,age,gender,tgl_masuk.strftime("%Y-%m-%d"),
                     tgl_periksa1.strftime("%Y-%m-%d"),exam_type,p,val,"", "DBD"])
    
    # Chemistry tests (some different day)
    chem_params = ["Albumin","AST","ALT","ALP","Bilirubin","LDL","HDL","Trigliserida"]
    
    for p in random.sample(chem_params, random.randint(3,6)):
        if p=="Albumin": val = albumin
        elif p=="AST": val = random.randint(40,120)
        elif p=="ALT": val = random.randint(40,110)
        elif p=="ALP": val = random.randint(70,140)
        elif p=="Bilirubin": val = round(random.uniform(0.8,2.0),1)
        elif p=="LDL": val = random.randint(90,160)
        elif p=="HDL": val = random.randint(35,60)
        elif p=="Trigliserida": val = random.randint(100,250)
        
        rows.append([no_rm,name,age,gender,tgl_masuk.strftime("%Y-%m-%d"),
                     tgl_periksa2.strftime("%Y-%m-%d"),"Kimia Darah",p,val,"","DBD"])

# Create DataFrame
df = pd.DataFrame(rows, columns=[
    "No_RM","Nama","Umur","Jenis_Kelamin","Tanggal_Masuk",
    "Tanggal_Pemeriksaan","Jenis_Pemeriksaan","Parameter","Hasil","Satuan","Diagnosa"
])

# Save file
file_path = "/mnt/data/data_rekam_medis_dbd_300_long_format_standar.csv"
df.to_csv(file_path, index=False)
file_path