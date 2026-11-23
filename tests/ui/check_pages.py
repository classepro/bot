import sys, traceback
from streamlit.testing.v1 import AppTest

PAGES = ["📊 Dashboard", "🎯 Prédiction", "🗂 Historique", "🧠 Entraînement",
         "🔬 Backtesting", "🗄 Données", "📄 Rapport scientifique"]
ok = True
for page in PAGES:
    try:
        at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "crashpredict/ui/app.py"), default_timeout=300)
        at.run()
        at.sidebar.radio[0].set_value(page).run()
        exc = [e.value[:400] for e in at.exception]
        errs = [e.value[:300] for e in at.error]
        print(f"{page:28} exceptions={len(exc)} errors={len(errs)}")
        for e in exc + errs:
            ok = False
            print("   ", e)
    except Exception:
        ok = False
        print(page, "CRASH"); traceback.print_exc()
print("ALL OK" if ok else "ISSUES")
