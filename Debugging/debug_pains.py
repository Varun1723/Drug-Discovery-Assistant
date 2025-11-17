import logging
from rdkit import Chem
from rdkit.Chem import FilterCatalog

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("--- [START] PAINS Filter Debug Test ---")

# --- 1. The molecules that SHOULD FAIL ---
pains_molecules = {
    "Catechol (Hydroquinone)": "C1=CC=C(C=C1O)O",
    "Rhodanine": "C1C(=S)NC(=O)S1",
    "Aflatoxin B1": "O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5"
}

# --- 2. Create the PAINS catalog (The 100% correct way) ---
logger.info("Initializing PAINS filter catalog...")
params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)

catalog = FilterCatalog.FilterCatalog(params)
logger.info(f"Catalog created. Total filter entries: {catalog.GetNumEntries()}")

if catalog.GetNumEntries() == 0:
    logger.error("CRITICAL FAILURE: Catalog is empty. RDKit is not loading PAINS filters.")
    exit()

print("--- [RUNNING] Testing Molecules ---")

all_tests_passed = True
for name, smiles in pains_molecules.items():
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print(f"Test for '{name}': ❌ FAILED (Invalid SMILES)")
        all_tests_passed = False
        continue
    
    entry = catalog.GetFirstMatch(mol)
    
    if entry:
        # This is the correct "FAIL" result we want
        print(f"Test for '{name}': ✅ SUCCESS. Correctly flagged as PAINS. (Filter: {entry.GetDescription()})")
    else:
        # This is the "PASS" result (our bug)
        print(f"Test for '{name}': ❌ FAILED. Incorrectly reported as 'Pass'.")
        all_tests_passed = False

print("--- [END] Test Complete ---")
if all_tests_passed:
    print("✅✅✅ Result: The RDKit PAINS filter is working perfectly in your environment.")
else:
    print("❌❌❌ Result: The RDKit PAINS filter is NOT working in your environment.")