TEST_QUERIES = [
    (
        "I have acidity and constipation, is there something that helps digestion?",
        ["alsactil", "digestion", "constipation", "acidity"],
        "EN | GI symptoms (acidity + constipation)",
    ),
    (
        "عندي حموضة وإمساك، هل هناك شيء يساعد على الهضم؟",
        ["alsactil", "الهضم", "الإمساك", "الحموضة"],
        "AR | GI symptoms (acidity + constipation)",
    ),
    (
        "My doctor said I have iron deficiency anemia during pregnancy, what supplement should I take?",
        ["amyron", "iron", "pregnancy", "hemoglobin"],
        "EN | Iron deficiency in pregnancy",
    ),
    (
        "طبيبتي قالت عندي فقر دم بسبب نقص الحديد وأنا حامل",
        ["amyron", "الحديد", "الحمل", "هيموجلوبين"],
        "AR | Iron deficiency in pregnancy",
    ),
    (
        "What can I take to reduce acne and purify my blood naturally?",
        ["neemol", "acne", "blood", "skin"],
        "EN | Acne & blood purification",
    ),
    (
        "أريد علاجًا طبيعيًا لحب الشباب وتنقية الدم",
        ["neemol", "حب الشباب", "الدم", "الجلد"],
        "AR | Acne & blood purification",
    ),
    (
        "I am underweight and want to gain muscle mass and increase appetite",
        ["aswagandhadi", "weight", "muscle", "appetite"],
        "EN | Weight gain & muscle",
    ),
    (
        "أنا نحيف وأريد زيادة الوزن وبناء العضلات",
        ["aswagandhadi", "وزن", "عضل", "شهية"],
        "AR | Weight gain & muscle",
    ),
    (
        "What is the dose of Brihat Vasavaleh for a 4-year-old child?",
        ["brihat", "children", "dose", "1-2"],
        "EN | Pediatric dosage (under 5)",
    ),
    (
        "ما الجرعة المناسبة لطفل عمره 4 سنوات من Brihat Vasavaleh؟",
        ["brihat", "أطفال", "جرعة", "1-2"],
        "AR | Pediatric dosage (under 5)",
    ),
    (
        "Which herbal tablet is not suitable for people allergic to Triphala?",
        ["alsactil", "triphala", "allerg"],
        "EN | Paraphrase — Triphala allergy",
    ),
    (
        "Can I chew the Ayurvedic tablets or do I need to swallow them whole?",
        ["alsactil", "chew", "crush", "powder"],
        "EN | Paraphrase — tablet form",
    ),
    (
        "Is there any Ayurvedic product that should NOT be taken during pregnancy?",
        ["rasna", "neemol", "pregnancy", "avoid"],
        "EN | Negation — pregnancy contraindication",
    ),
    (
        "Which products contain sugar and are therefore unsafe for diabetics?",
        ["aswagandhadi", "ashwagandha avaleha", "sugar", "diabetic"],
        "EN | Negation — sugar / diabetics",
    ),
    (
        "What should I eat or change in my diet if I have insulin resistance?",
        ["insulin", "diet", "exercise", "diabetes"],
        "EN | Insulin resistance management",
    ),
    (
        "ماذا أفعل إذا كان عندي مقاومة للأنسولين؟",
        ["insulin", "إنسولين", "نظام", "غذائي"],
        "AR | Insulin resistance management",
    ),
    (
        "How many years of clinical experience does the Mumbai nutritionist have?",
        ["sayantani", "21", "22", "mumbai"],
        "EN | Credential lookup",
    ),
    (
        "What is the boiling point of sulfuric acid?",
        [],
        "EN | Out-of-domain (chemistry)",
    ),
    (
        "Tell me the latest football scores from last weekend",
        [],
        "EN | Out-of-domain (sports)",
    ),
]
