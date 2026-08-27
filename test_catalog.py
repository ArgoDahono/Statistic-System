software_catalog = {
    'JASP': {
        'Correlation': ['Pearson Correlation', 'Spearman Correlation'],
        'T-Tests': ['One-Sample T-Test', 'Independent Samples T-Test']
    },
    'SPSS': {
        'Correlation': ['Pearson Correlation', 'Spearman Correlation'],
        'T-Tests': ['One-Sample T-Test', 'Independent Samples T-Test']
    },
    'jamovi': {
        'Correlation': ['Correlation Matrix'],
        'T-Tests': ['T-Test']
    }
}

print("Type:", type(software_catalog))
print("Keys:", list(software_catalog.keys()))