import pandas as pd
import numpy as np

# Load the data
file_path = r'd:\Data E\poltekkes\Project Zero PL.73.37-Sparta (SKRIPSI HEMA)\Skripsi hematologi-kimia klinik\Phyton\data_rekam_medis_dbd_300_long_format_standar.csv'
df = pd.read_csv(file_path)

print("=" * 80)
print("ANALISIS DUPLIKAT NOMER REKAM MEDIS")
print("=" * 80)
print(f"\nTotal baris data: {len(df)}")
print(f"Total Nomer RM unik: {df['No_RM'].nunique()}")

# Get unique records (since data is in long format, each patient appears multiple times)
unique_patients = df.drop_duplicates(subset=['No_RM', 'Nama', 'Umur'])
print(f"Total pasien unik (No_RM + Nama + Umur): {len(unique_patients)}")

print("\n" + "=" * 80)
print("1. ANALISIS: Apakah ada No_RM yang memiliki NAMA BERBEDA?")
print("=" * 80)

rm_names = unique_patients.groupby('No_RM')['Nama'].unique()
duplicate_names = rm_names[rm_names.apply(len) > 1]

if len(duplicate_names) > 0:
    print(f"\n⚠️  DITEMUKAN {len(duplicate_names)} No_RM dengan nama berbeda:\n")
    for rm, names in duplicate_names.items():
        print(f"  No_RM: {rm}")
        print(f"    Nama-nama yang terkait: {names}")
        # Show details
        patients = unique_patients[unique_patients['No_RM'] == rm][['No_RM', 'Nama', 'Umur', 'Jenis_Kelamin', 'Tanggal_Masuk']]
        print(f"    Detail:")
        for idx, row in patients.iterrows():
            print(f"      - {row['Nama']} (Umur: {row['Umur']}, Jenis Kelamin: {row['Jenis_Kelamin']}, Tgl Masuk: {row['Tanggal_Masuk']})")
        print()
else:
    print("\n✓ TIDAK ada No_RM dengan nama berbeda")

print("\n" + "=" * 80)
print("2. ANALISIS: Apakah ada No_RM yang memiliki UMUR BERBEDA?")
print("=" * 80)

rm_ages = unique_patients.groupby('No_RM')['Umur'].unique()
duplicate_ages = rm_ages[rm_ages.apply(len) > 1]

if len(duplicate_ages) > 0:
    print(f"\n⚠️  DITEMUKAN {len(duplicate_ages)} No_RM dengan umur berbeda:\n")
    for rm, ages in duplicate_ages.items():
        print(f"  No_RM: {rm}")
        print(f"    Umur-umur yang terkait: {sorted(ages)}")
        # Show details
        patients = unique_patients[unique_patients['No_RM'] == rm][['No_RM', 'Nama', 'Umur', 'Jenis_Kelamin', 'Tanggal_Masuk']]
        print(f"    Detail:")
        for idx, row in patients.iterrows():
            print(f"      - {row['Nama']} (Umur: {row['Umur']}, Jenis Kelamin: {row['Jenis_Kelamin']}, Tgl Masuk: {row['Tanggal_Masuk']})")
        print()
else:
    print("\n✓ TIDAK ada No_RM dengan umur berbeda")

print("\n" + "=" * 80)
print("3. ANALISIS: Apakah ada NAMA + UMUR yang memiliki No_RM BERBEDA?")
print("=" * 80)

# Group by Nama + Umur to find if same people have different RM numbers
name_age_groups = unique_patients.groupby(['Nama', 'Umur'])['No_RM'].unique()
duplicate_rm_for_same_person = name_age_groups[name_age_groups.apply(len) > 1]

if len(duplicate_rm_for_same_person) > 0:
    print(f"\n⚠️  DITEMUKAN {len(duplicate_rm_for_same_person)} orang (Nama + Umur sama) dengan No_RM berbeda:\n")
    for (name, age), rm_list in duplicate_rm_for_same_person.items():
        print(f"  Nama: {name}, Umur: {age}")
        print(f"    No_RM yang terkait: {sorted(rm_list)}")
        # Show details
        patients = unique_patients[(unique_patients['Nama'] == name) & (unique_patients['Umur'] == age)][['No_RM', 'Nama', 'Umur', 'Jenis_Kelamin', 'Tanggal_Masuk']]
        print(f"    Detail:")
        for idx, row in patients.iterrows():
            print(f"      - No_RM: {row['No_RM']} (Jenis Kelamin: {row['Jenis_Kelamin']}, Tgl Masuk: {row['Tanggal_Masuk']})")
        print()
else:
    print("\n✓ TIDAK ada orang (Nama + Umur sama) dengan No_RM berbeda")

print("\n" + "=" * 80)
print("4. SUMMARY STATISTIK DUPLIKAT")
print("=" * 80)

# Check for duplicate No_RM (multiple times) in the dataset
rm_counts = unique_patients['No_RM'].value_counts()
duplicate_rm = rm_counts[rm_counts > 1]

if len(duplicate_rm) > 0:
    print(f"\nNo_RM yang muncul lebih dari satu kali di dataset unik: {len(duplicate_rm)}")
    print("(Ini bisa terjadi karena kombinasi nama/umur berbeda untuk RM yang sama)")
    print(duplicate_rm.head(10))
else:
    print("\n✓ Setiap No_RM unik hanya muncul satu kali")

print("\n" + "=" * 80)
print("KESIMPULAN")
print("=" * 80)

total_issues = len(duplicate_names) + len(duplicate_ages) + len(duplicate_rm_for_same_person)
print(f"\nTotal isu duplikat yang ditemukan: {total_issues}")
if total_issues == 0:
    print("✓ DATA CLEAN - Tidak ada duplikat problematik ditemukan")
else:
    print("⚠️  Ada isu duplikat potensial yang perlu diverifikasi")
