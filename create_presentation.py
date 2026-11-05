# -*- coding: utf-8 -*-
"""
scMamba2 + BiMamba Proje Tasarım Dersi Ara Raporu Sunum Üreteci (.pptx)
Tasarım ve Bitirme Projesi - Ara Rapor Sunumu
Öğrenci: Mithat Yasin Çeliktaş
"""

import os
from pathlib import Path
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_path: str):
    prs = Presentation()
    # 16:9 Geniş Ekran Standardı (12192000 x 6858000 EMU)
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    FONT_FAMILY = "Segoe UI"

    # Renk Paleti (Akademik / Profesyonel Medikal Bilişim Teması)
    C_NAVY_DARK = RGBColor(15, 23, 42)      # #0F172A (Koyu Başlık / Kapak Arka Planı)
    C_NAVY_MED  = RGBColor(30, 41, 59)      # #1E293B
    C_BLUE      = RGBColor(2, 132, 199)     # #0284C7 (Vurgu Mavisi)
    C_TEAL      = RGBColor(16, 185, 129)    # #10B981 (Başarı Yeşili)
    C_PURPLE    = RGBColor(124, 58, 237)    # #7C3AED (Biyoloji Vurgusu)
    C_RED       = RGBColor(239, 68, 68)     # #EF4444 (Kanser Vurgusu)
    C_GRAY_BG   = RGBColor(248, 250, 252)   # #F8FAFC (Slayt Arka Planı)
    C_CARD_BG   = RGBColor(255, 255, 255)   # #FFFFFF (Kart Arka Planı)
    C_BORDER    = RGBColor(226, 232, 240)   # #E2E8F0 (Kart Kenarlığı)
    C_TEXT_DARK = RGBColor(15, 23, 42)      # #0F172A
    C_TEXT_MUTED= RGBColor(100, 116, 139)   # #64748B
    C_TEXT_LIGHT= RGBColor(255, 255, 255)
    C_PH_BG     = RGBColor(241, 245, 249)   # #F1F5F9 (Placeholder Arka Planı)
    C_PH_BORDER = RGBColor(148, 163, 184)   # #94A3B8 (Placeholder Çerçevesi)

    def set_slide_bg(slide, color):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_run_text(paragraph, text, size_pt, color=C_TEXT_DARK, bold=False, italic=False):
        run = paragraph.add_run()
        run.text = text
        run.font.name = FONT_FAMILY
        run.font.size = Pt(size_pt)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
        return run

    def add_header(slide, title, category="PROJE TASARIM DERSI ARA RAPORU SUNUMU", slide_num=None):
        set_slide_bg(slide, C_GRAY_BG)

        # Üst İnce Vurgu Çizgisi
        accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.08))
        accent.fill.solid()
        accent.fill.fore_color.rgb = C_BLUE
        accent.line.fill.background()

        # Kategori & Sayfa Başlığı
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.7), Inches(0.95))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p0 = tf.paragraphs[0]
        add_run_text(p0, category.upper(), 9.5, color=C_BLUE, bold=True)

        p1 = tf.add_paragraph()
        add_run_text(p1, title, 19, color=C_NAVY_DARK, bold=True)

        # Sayfa Numarası & Footer
        if slide_num:
            ftr = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(11.7), Inches(0.35))
            ftf = ftr.text_frame
            ftf.margin_left = ftf.margin_top = ftf.margin_right = ftf.margin_bottom = 0
            fp = ftf.paragraphs[0]
            add_run_text(fp, f"Mithat Yasin Celiktas | scMamba2 + BiMamba Hiyerarsik Klinik Teshis ve XAI Sistemi", 8.5, color=C_TEXT_MUTED)
            add_run_text(fp, f"                                                                               Slayt {slide_num} / 15", 8.5, color=C_TEXT_MUTED, bold=True)

    def add_card(slide, left, top, width, height, title=None, subtitle=None, bg_color=C_CARD_BG, border_color=C_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.0)

        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.18), width - Inches(0.4), Inches(0.6))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            add_run_text(p, title, 13, color=C_NAVY_DARK, bold=True)

            if subtitle:
                p2 = tf.add_paragraph()
                add_run_text(p2, subtitle, 9.5, color=C_TEXT_MUTED)
        return card

    def add_placeholder(slide, left, top, width, height, placeholder_title, subtitle="Isı Haritası / Grafik Görseli"):
        box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
        box.fill.solid()
        box.fill.fore_color.rgb = C_PH_BG
        box.line.color.rgb = C_PH_BORDER
        box.line.width = Pt(1.5)

        tb_top = top + (height // 2) - Inches(0.7)
        tb = slide.shapes.add_textbox(left + Inches(0.2), tb_top, width - Inches(0.4), Inches(1.4))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p1 = tf.paragraphs[0]
        p1.alignment = PP_ALIGN.CENTER
        add_run_text(p1, "[ GORSEL / ISI HARITASI ALANI ]", 12, color=C_BLUE, bold=True)

        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        add_run_text(p2, placeholder_title, 11, color=C_NAVY_DARK, bold=True)

        p3 = tf.add_paragraph()
        p3.alignment = PP_ALIGN.CENTER
        add_run_text(p3, f"({subtitle})", 9, color=C_TEXT_MUTED, italic=True)
        return box

    def add_stat_box(slide, left, top, width, height, stat_value, stat_label, accent_color=C_BLUE):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = C_CARD_BG
        card.line.color.rgb = C_BORDER
        card.line.width = Pt(1.0)

        # Üst renkli çizgi
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Inches(0.06))
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent_color
        bar.line.fill.background()

        tb = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.15), width - Inches(0.3), height - Inches(0.2))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        add_run_text(p, stat_value, 22, color=accent_color, bold=True)

        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        add_run_text(p2, stat_label, 9.5, color=C_NAVY_DARK, bold=True)

    def add_bullet_points(slide, left, top, width, height, points):
        tb = slide.shapes.add_textbox(left, top, width, height)
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        for i, pt_text in enumerate(points):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            add_run_text(p, pt_text, 10.5, color=C_NAVY_DARK)
            p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 1: BAŞLIK / KAPAK SLAYTI (Dark Premium Theme)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1, C_NAVY_DARK)

    deco = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(7.35), prs.slide_width, Inches(0.15))
    deco.fill.solid()
    deco.fill.fore_color.rgb = C_BLUE
    deco.line.fill.background()

    # Üst Rozet
    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.1), Inches(11.3), Inches(0.5))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    add_run_text(p, "BILGISAYAR MUHENDISLIGI BOLUMU  |  TASARIM VE BITIRME PROJESI - ARA RAPOR SUNUMU", 10, color=C_BLUE, bold=True)

    # Ana Başlık
    tb_title = s1.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(11.3), Inches(2.2))
    tft = tb_title.text_frame
    tft.word_wrap = True
    pt = tft.paragraphs[0]
    add_run_text(pt, "scMamba2 + BiMamba ile Hiyerarsik Klinik Teshis ve Aciklanabilir Yapay Zeka (XAI) Karar Destek Sistemi", 28, color=C_TEXT_LIGHT, bold=True)

    pt_sub = tft.add_paragraph()
    add_run_text(pt_sub, "Tek Hucreli (scRNA-seq) ve Toplu Doku (Bulk RNA-seq) Seviyesinde Kanser ve Kanser Kok Hucresi (CSC) Tespiti", 14, color=RGBColor(148, 163, 184))
    pt_sub.space_before = Pt(8)

    # Bilgi Kutusu Kartı
    info_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.3), Inches(11.33), Inches(2.3))
    info_card.fill.solid()
    info_card.fill.fore_color.rgb = C_NAVY_MED
    info_card.line.color.rgb = RGBColor(51, 65, 85)

    tb_meta = s1.shapes.add_textbox(Inches(1.4), Inches(4.55), Inches(10.5), Inches(1.8))
    tfm = tb_meta.text_frame
    tfm.word_wrap = True

    p = tfm.paragraphs[0]
    add_run_text(p, "[Proje Yurutucusu]: Mithat Yasin Celiktas", 13, color=C_TEXT_LIGHT, bold=True)

    p2 = tfm.add_paragraph()
    add_run_text(p2, "[Danisman / Ders]: Tasarim ve Bitirme Projesi Ara Rapor Degerlendirme Jurisi", 11.5, color=RGBColor(203, 213, 225))
    p2.space_before = Pt(6)

    p3 = tfm.add_paragraph()
    add_run_text(p3, "[GitHub Deposu]: https://github.com/MithatYasinCeliktas/tasarimProje", 10.5, color=C_BLUE)
    p3.space_before = Pt(6)

    p4 = tfm.add_paragraph()
    add_run_text(p4, "[Ar-Ge Test Tezgahi]: Streamlit Web Platformu (Port: 8501) | Kaggle 2x Tesla T4 GPU Hibrit Koprusu", 10.5, color=C_TEAL)
    p4.space_before = Pt(4)


    # =========================================================================
    # SLIDE 2: KLİNİK MOTİVASYON VE PROBLEM TANIMI
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Klinik Problem Tanimi ve Biyolojik Motivasyon", slide_num=2)

    add_card(s2, Inches(0.8), Inches(1.45), Inches(5.6), Inches(4.2),
             title="Biyolojik Problem: Tumor Heterojenitesi ve CSC",
             subtitle="Geleneksel Biyopsilerin ve Doku Ortalamasinin Yetersizligi")
    add_bullet_points(s2, Inches(1.05), Inches(2.25), Inches(5.1), Inches(3.2), [
        "- Bulk RNA-seq Maskelemesi: Klasik doku biyopsileri milyonlarca hucrenin ortalamasini alir; nadir ancak en olumcul alt populasyonlari gizler.",
        "- Kanser Kok Hucreleri (CSC): Kemoterapi ve radyoterapiye direncli, nuksetme ve metastazi baslatan asil kok hucre grubudur.",
        "- Klinik Ihtiyac: Tek tek hucre duzeyinde ve doku karisimlarinda kok hucre sinyallerini (CD44, ALDH1A1, PROM1) ayirt edebilecek hassas bir tani motoru sarttir."
    ])

    add_card(s2, Inches(6.8), Inches(1.45), Inches(5.7), Inches(4.2),
             title="Bilisimsel Zorluk: 60,000+ Gen ve Dropout Gurultusu",
             subtitle="Neden Transformer Mimarileri Bu Olcekte Tíkaniyor?")
    add_bullet_points(s2, Inches(7.05), Inches(2.25), Inches(5.2), Inches(3.2), [
        "- Asiri Yuksek Boyutluluk: Insan genomunda 60,700 gen bulunur. Hucre basina seyreklik (sparsity/dropout) %85-90 duzeyindedir.",
        "- Transformer Darbogazi O(N2): Self-Attention mekanizmasi 512 genin otesinde bellek tuketimini katlar; GPU'larda OOM olusturur.",
        "- Cozum: Lineer O(N) karmasiklikla calisan State-Space Model (SSM) Mamba mimarisi ve cift yonlu BiMamba etkilesim katmani."
    ])

    add_card(s2, Inches(0.8), Inches(5.85), Inches(11.7), Inches(1.05), bg_color=C_NAVY_DARK, border_color=C_BLUE)
    tb = s2.shapes.add_textbox(Inches(1.0), Inches(5.95), Inches(11.3), Inches(0.85))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    add_run_text(p, "Temel Hedef: scMamba2 + BiMamba mimarisi ile hem tek hucrede hem de toplu dokuda 6 kademeli hiyerarsik klinik teshis koymak ve kararlari gen seviyesinde aciklanabilir kilmaktir.", 11.5, color=C_TEXT_LIGHT, bold=True)


    # =========================================================================
    # SLIDE 3: PROJENİN AMACI VE ÖZGÜN DEĞERİ
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Projenin Amaci, Kapsami ve Ozgun Degeri (Novelty)", slide_num=3)

    add_card(s3, Inches(0.8), Inches(1.5), Inches(3.7), Inches(5.2),
             title="1. scMamba2 + BiMamba Hibriti",
             subtitle="Foundation Model + Cift Yonlu SSM")
    add_bullet_points(s3, Inches(1.0), Inches(2.35), Inches(3.3), Inches(4.1), [
        "- 60,700 Gen Sozlugu: scMamba2 on egitimli temel omurga ile hucrenin global temsili cikarilir.",
        "- Cift Yonlu BiMamba: Genomik dizilimdeki ileri ve geri gen duzenleme etkilesimleri modellenir.",
        "- Lineer O(N) Verimlilik: Transformer'lara kiyasla 4x daha hizli cikarim ve dusuk bellek tuketimi."
    ])

    add_card(s3, Inches(4.8), Inches(1.5), Inches(3.7), Inches(5.2),
             title="2. 6 Kademeli Hiyerarsik Teshis",
             subtitle="Tek Asamali Siniflandiricilar Yerine Agac Yapisi")
    add_bullet_points(s3, Inches(5.0), Inches(2.35), Inches(3.3), Inches(4.1), [
        "- Duz multiclass yerine onkolojik karar agaci simule edilir.",
        "- 1. Hucre Tipi (14 Sinif)",
        "- 4a. Saglik Durumu (Normal vs Hasta)",
        "- 4b. Hastalik Arketipi (COVID, MS, Kanser)",
        "- 4c. Malignite / Kanser Tespiti",
        "- 4d. 11 Kanser Varyanti (Organ Kokeni)",
        "- 4d-Ek. Kanser Kok Hucresi (CSC) Varligi"
    ])

    add_card(s3, Inches(8.8), Inches(1.5), Inches(3.7), Inches(5.2),
             title="3. Aciklanabilir XAI ve Isı Haritalari",
             subtitle="Kara Kutu Yapay Zekayi Kirma")
    add_bullet_points(s3, Inches(9.0), Inches(2.35), Inches(3.3), Inches(4.1), [
        "- Input x Gradient Atifi: Karari belirleyen Top-18 kilit gen hekime gosterilir.",
        "- 16 Biyolojik Yolak Skoru: Kok hucre, EMT metastaz ve hipoksi aktivasyonu olculur.",
        "- 4-Panelli Master Heatmap: Guven, gen, varyant ve yolak seviyesinde tek parca tani raporu."
    ])


    # =========================================================================
    # SLIDE 4: SİSTEM MİMARİSİ VE AKIŞ ŞEMASI
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Sistem Mimarisi ve Uctan Uca Veri Akisi", slide_num=4)

    add_card(s4, Inches(0.8), Inches(1.45), Inches(5.3), Inches(5.4),
             title="Mimari Bilesenleri ve Tensor Boyutlari",
             subtitle="scMamba2 + BiMamba Entegrasyonu")
    add_bullet_points(s4, Inches(1.05), Inches(2.25), Inches(4.8), Inches(4.4), [
        "1. Girdi Katmani (AnnData / Tensor):",
        "   - tokens_genes (B, 1024) & tokens_vals (B, 1024)",
        "   - panel_expr (B, 512): Kromozom sirali Master Genler",
        "   - bio_sigs_norm (B, 16): Biyolojik Yolak Z-skorlari",
        "2. scMamba2 Backbone (Pretrained 512-d Temsil):",
        "   - ContinuousValueEncoder + State-Space Bloklari",
        "3. Gated BiMamba Karar Motoru (2 Katmanli):",
        "   - Ileri SSM + Geri SSM + Kapi Fuzyon Agi",
        "4. Cok Gorevli Kayip Fonksiyonu (Multi-Task Loss):",
        "   - Focal Loss (Dengesiz Kanser Varyantlari)",
        "   - CSC Weighted BCE & Cosine Prototip Kaybi",
        "   - ECE Kalibrasyon Cezalandirmasi"
    ])

    add_placeholder(s4, Inches(6.4), Inches(1.45), Inches(6.1), Inches(5.4),
                    placeholder_title="SISTEM VE MODEL MIMARISI AKIS SEMASI",
                    subtitle="scMamba2 Backbone -> Gated BiMamba Head -> 6 Teshis Basligi")


    # =========================================================================
    # SLIDE 5: 6 KADEMELİ HİYERARŞİK KLİNİK TEŞHİS AĞACI
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "6 Kademeli Hiyerarsik Klinik Karar Agaci", slide_num=5)

    levels = [
        ("Seviye 1: Hucre Tipi (Cell Type)", "14 Temel Biyolojik Sinif",
         "Malign Kanser Hucresi, T Lenfosit, B Lenfosit, Makrofaj/Monosit, Epitelyal Hucre, Noron, Astrosit, Oligodendrosit, Fibroblast, Endotel vb.", C_BLUE),
        ("Seviye 4a: Genel Saglik Durumu", "2 Sinif: Saglikli vs. Hastalikli",
         "Saglikli bir donorden mi yoksa patolojik inflamasyon/tumor tasiyan bir hastadan mi geldigini dogrular.", C_TEAL),
        ("Seviye 4b: Hastalik Arketipi", "4 Ana Klinik Kategori",
         "1. Saglikli Kontrol | 2. Akut Viral Enfeksiyon (COVID-19) | 3. Otoimmun Noro-Inflamasyon (MS) | 4. Kanser / Malignite", C_PURPLE),
        ("Seviye 4c: Malignite / Kanser Tespiti", "2 Sinif: Kanser Degil vs. Malign Kanser",
         "Hastaligin kesin onkolojik tumor hucresi olup olmadigini %99.95 dogrulukla belirler.", C_RED),
        ("Seviye 4d: 11 Kanser Varyanti (Organ Kokeni)", "11 Pan-Kanser Molekuler Varyanti",
         "Non-Cancer, Akciger (LUAD), A549, Kronik Miyeloid Losemi (CML), Bobrek (RCC), Rahim (UCEC), Ozofagus (ESCA), Miyelom (MYE), Tiroid (THCA), Yumurtalik (OV), Pankreas (PAAD).", C_BLUE),
        ("Seviye 4d-Ek: Kanser Kok Hucresi (CSC)", "2 Sinif: CSC- Negatif vs. CSC+ POZITIF",
         "CD44, ALDH1A1 ve CytoTRACE entropisi ile nuks ve kemoterapi direnci potansiyeli yuksek kok hucre karakterini (%98.16 dogruluk) saptar.", C_RED),
    ]

    for idx, (l_title, l_sub, l_desc, l_col) in enumerate(levels):
        r = idx // 2
        c = idx % 2
        l_x = Inches(0.8) + c * Inches(5.95)
        l_y = Inches(1.45) + r * Inches(1.75)

        add_card(s5, l_x, l_y, Inches(5.75), Inches(1.6), title=l_title, subtitle=l_sub)
        bar = s5.shapes.add_shape(MSO_SHAPE.RECTANGLE, l_x, l_y, Inches(0.08), Inches(1.6))
        bar.fill.solid()
        bar.fill.fore_color.rgb = l_col
        bar.line.fill.background()

        tb = s5.shapes.add_textbox(l_x + Inches(0.2), l_y + Inches(0.75), Inches(5.35), Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        add_run_text(p, l_desc, 9.5, color=C_NAVY_DARK)


    # =========================================================================
    # SLIDE 6: VERİ SETİ VE ÖZELLİK MÜHENDİSLİĞİ
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Cok Dokulu Birlesik Veri Seti ve Ozellik Muhendisligi", slide_num=6)

    add_stat_box(s6, Inches(0.8), Inches(1.45), Inches(2.7), Inches(1.2), "38,263", "Toplam Hucre ve Doku", C_BLUE)
    add_stat_box(s6, Inches(3.8), Inches(1.45), Inches(2.7), Inches(1.2), "8,486", "Gorulmemis Test Hucresi", C_TEAL)
    add_stat_box(s6, Inches(6.8), Inches(1.45), Inches(2.7), Inches(1.2), "512", "Master Tani Gen Paneli", C_PURPLE)
    add_stat_box(s6, Inches(9.8), Inches(1.45), Inches(2.7), Inches(1.2), "16", "Biyolojik Yolak Skoru", C_RED)

    add_card(s6, Inches(0.8), Inches(2.85), Inches(5.7), Inches(4.0),
             title="Kati Veri Bolunmesi (No Data-Leakage)",
             subtitle="Egitim, Dogrulama ve Dis Test Kumeleri")
    add_bullet_points(s6, Inches(1.05), Inches(3.6), Inches(5.2), Inches(3.1), [
        "- Egitim Seti (Train): 25,791 Ornek (%67.4)",
        "- Dogrulama Seti (Validation): 3,986 Ornek (%10.4) - Erken durdurma ve hiperparametre optimizasyonu.",
        "- Dis Test Seti (Unseen Test): 8,486 Ornek (%22.2) - Modelin egitimde hic gormedigi hucre ve hastalar.",
        "- Modalite Dagilimi: 33,943 tekil hucre (scRNA-seq) + 4,320 Dirichlet agirlikli sentetik biyopsi (Bulk RNA-seq)."
    ])

    add_card(s6, Inches(6.8), Inches(2.85), Inches(5.7), Inches(4.0),
             title="16 Biyolojik Yolak Imzasi (Feature Engineering)",
             subtitle="Molekuler Sapmalari Ozetleyen Z-Skorlari")
    add_bullet_points(s6, Inches(7.05), Inches(3.6), Inches(5.2), Inches(3.1), [
        "- Kanser ve Kok Hucre: CSC Stemness, Proliferasyon (MKI67/TOP2A), EMT Metastaz (VIM/SNAI1), Hipoksi (HIF1A/LDHA).",
        "- Immun ve Inflamasyon: Viral Yanit (ISG15/IFIT1), MS Noro-Inflamasyon (GFAP/APOE), Makrofaj SPP1 & C1QC Imzalari, T-Hucre Tukenmisligi.",
        "- Soy ve Kararsizlik: Epitelyal, Noral/Glial, Hematopoietik soylar, CytoTRACE Kok Hucre Entropisi ve Kromozomal CNV Kararsizligi."
    ])


    # =========================================================================
    # SLIDE 7: AÇIKLANABİLİR YAPAY ZEKA (XAI) VE INPUT X GRADIENT
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Aciklanabilir Yapay Zeka (XAI) ve Karar Gerekcelendirme", slide_num=7)

    add_card(s7, Inches(0.8), Inches(1.45), Inches(5.5), Inches(5.4),
             title="Kara Kutu Engeli Nasil Asildi?",
             subtitle="Klinik Dogrulama Icin Matematiksel Atif")
    add_bullet_points(s7, Inches(1.05), Inches(2.3), Inches(5.0), Inches(4.3), [
        "1. BiMamba Input x Gradient Atifi:",
        "   Modelin karar ciktilarindan geriye dogru gradyan akitilir. Her genin gradyan buyuklugu ile ifade seviyesi carpilarak salience skoru hesaplanir.",
        "   Importance(g) = |x * dLoss/dx| + 0.35 * (x / x_max)",
        "2. Top-18 Biyobelirtec Gen Secimi:",
        "   Karara en yuksek etkiyi yapan ilk 18 gen anlik olarak siralanir ve saglikli kontrol referansi ile karsilastirilir.",
        "3. Kanser Prototip Benzerlik Matrisi:",
        "   Gizli uzaydaki 11 kanser prototipi ile hastanin hucresi arasindaki Kosinus Benzerligi [0, 100] araliginda olculur.",
        "4. Biyolojik Anotasyon Sozlugu:",
        "   Model, her genin klinik fonksiyonunu (or. CD44: Kok Hucre, MPO: Miyeloid Losemi) hekimin anlayacagi dilde raporlar."
    ])

    add_placeholder(s7, Inches(6.6), Inches(1.45), Inches(5.9), Inches(5.4),
                    placeholder_title="XAI KARAR AGIRLIKLARI VE PROTOTIP UYUM GRAFIGI",
                    subtitle="Input x Gradient Gen Siralamasi & Kosinus Benzerlik Cubuklari")


    # =========================================================================
    # SLIDE 8: 4-PANELLİ DİNAMİK TEŞHİS RAPORU (MASTER POSTER)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "4-Panelli Dinamik Klinik Teshis Raporu (Genel Bakis)", slide_num=8)

    add_card(s8, Inches(0.8), Inches(1.45), Inches(4.5), Inches(5.4),
             title="Master Raporun 4 Katmani",
             subtitle="3000x2025 px Yuksek Cozunurluklu Cikti")
    add_bullet_points(s8, Inches(1.0), Inches(2.3), Inches(4.1), Inches(4.3), [
        "- Panel A (Sol-Ust): Hiyerarsik Klinik Karar Agaci & Kalibre Guven Yuzdeleri (%95 klinik hedef esigi).",
        "- Panel B (Sag-Ust): Karari Belirleyen Top-18 Biyobelirtec Geni ve XAI Isı Haritasi (Magma renk paleti).",
        "- Panel C (Sol-Alt): 11 Kanser Varyanti Prototip Oruntu Uyum Isı Haritasi (YlOrRd renk paleti).",
        "- Panel D (Sag-Alt): 16 Biyolojik Yolak, Kok Hucre (CSC) ve Kromozomal CNV Isı Haritasi (RdBu_r paleti).",
        "- Indirilebilir Format: Tek tikla HD PNG klinik arsiv ciktisi."
    ])

    add_placeholder(s8, Inches(5.6), Inches(1.45), Inches(6.9), Inches(5.4),
                    placeholder_title="4-PANELLI BIRLESIK MASTER TESHIS POSTERI (PANEL A + B + C + D)",
                    subtitle="latest_diagnostic_report.png | 3000x2025 px HD Grafik Alani")


    # =========================================================================
    # SLIDE 9: PANEL A VE PANEL B DETAYLI İNCELEMESİ
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Panel A (Teshis / Guven) & Panel B (Top-18 Gen Heatmap)", slide_num=9)

    add_card(s9, Inches(0.8), Inches(1.45), Inches(5.7), Inches(2.2),
             title="Panel A: Hiyerarsik Teshis ve Kalibre Guven Cubuklari",
             subtitle="Modelin 6 Kademedeki Karar Kesinligi")
    add_bullet_points(s9, Inches(1.0), Inches(2.25), Inches(5.3), Inches(1.3), [
        "- 6 asamanin tamami icin tahmin edilen sinif, gercek etiket ve guven yuzdesi.",
        "- %95 guven esigi referans cizgisi. Yesil cubuklar dogru, kirmizi cubuklar sapmayi gosterir."
    ])
    add_placeholder(s9, Inches(0.8), Inches(3.8), Inches(5.7), Inches(3.05),
                    placeholder_title="PANEL A: HIYERARSIK GUVEN CUBUKLARI GRAFIGI",
                    subtitle="fig_panel_a | Model Guven Dagilimi")

    add_card(s9, Inches(6.8), Inches(1.45), Inches(5.7), Inches(2.2),
             title="Panel B: Top-18 Biyobelirtec Gen Feature Heatmap'i",
             subtitle="Hasta Ifadesi vs. Saglikli ve Kanser Referanslari")
    add_bullet_points(s9, Inches(7.0), Inches(2.25), Inches(5.3), Inches(1.3), [
        "- Karara en cok etki eden ilk 18 gen (CD44, ALDH1A1, EPCAM, MKI67 vb.).",
        "- 4 Satirli Isı Haritasi: Hasta Ifadesi | Eslesen Ref | Saglikli Ref | BiMamba XAI Agirligi."
    ])
    add_placeholder(s9, Inches(6.8), Inches(3.8), Inches(5.7), Inches(3.05),
                    placeholder_title="PANEL B: TOP-18 GEN FEATURE & XAI ISI HARITASI",
                    subtitle="fig_panel_b | Magma Colormap Heatmap")


    # =========================================================================
    # SLIDE 10: PANEL C VE PANEL D DETAYLI İNCELEMESİ
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Panel C (Varyant Benzerligi) & Panel D (16 Biyolojik Yolak)", slide_num=10)

    add_card(s10, Inches(0.8), Inches(1.45), Inches(5.7), Inches(2.2),
             title="Panel C: 11 Kanser Varyanti Prototip Oruntu Uyumu",
             subtitle="Hangi Kansere Ne Derece Benziyor?")
    add_bullet_points(s10, Inches(1.0), Inches(2.25), Inches(5.3), Inches(1.3), [
        "- Hastanin hucre profilinin 11 kanser tipi ile kosinus benzerligi matrisi.",
        "- 2 Satir: Model Tahmin Olasiligi (%) ve Prototip Oruntu Benzerligi (%).",
        "- Metastatik ve primer ayriminda organ kokeni dogrulamasi saglar."
    ])
    add_placeholder(s10, Inches(0.8), Inches(3.8), Inches(5.7), Inches(3.05),
                    placeholder_title="PANEL C: 11 KANSER VARYANTI ORUNTU UYUM ISI HARITASI",
                    subtitle="fig_panel_c | YlOrRd Colormap Heatmap")

    add_card(s10, Inches(6.8), Inches(1.45), Inches(5.7), Inches(2.2),
             title="Panel D: 16 Biyolojik Yolak & Kok Hucre (CSC) Imzalari",
             subtitle="Fonksiyonel Hucresel Durum Sapmalari")
    add_bullet_points(s10, Inches(7.0), Inches(2.25), Inches(5.3), Inches(1.3), [
        "- 16 yolagin Z-skor sapmasi (+1.5 Z uzeri patolojik yuksek, -1.5 Z alti baskilanmis).",
        "- Kok Hucre (CSC), Proliferasyon, EMT Metastaz ve Viral ISG aktivasyon seviyeleri.",
        "- Ikinci satirda BiMamba'nin o yolaga verdigi karar onemi gosterilir."
    ])
    add_placeholder(s10, Inches(6.8), Inches(3.8), Inches(5.7), Inches(3.05),
                    placeholder_title="PANEL D: 16 YOLAK & CSC IMZA ISI HARITASI",
                    subtitle="fig_panel_d | RdBu_r Diverging Colormap Heatmap")


    # =========================================================================
    # SLIDE 11: BULUT ALTYAPISI - KAGGLE 2X T4 GPU KÖPRÜSÜ
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Hibrit Bulut Altyapisi: Kaggle 2x Tesla T4 GPU Koprusu", slide_num=11)

    add_card(s11, Inches(0.8), Inches(1.45), Inches(5.5), Inches(5.4),
             title="Sifir Maliyetli Hibrit Hesaplama Altyapisi",
             subtitle="Yerel Masaustu ile Kaggle 2x T4 Entegrasyonu")
    add_bullet_points(s11, Inches(1.05), Inches(2.3), Inches(5.0), Inches(4.3), [
        "- Kaggle GPU Daemon (2x Tesla T4, 32 GB VRAM):",
        "  Arka planda calisan Python HTTP sunucusu ve Cloudflare tuneli otomatik olarak ayaga kalkar.",
        "- Cloudflare HTTPS Tuneli:",
        "  Dinamik HTTPS tuneli ntfy.sh bildirim protokoluyle yerel bilgisayara baglanir.",
        "- Sifir Kota Israfi (Zero Quota Waste):",
        "  * 'python kaggle_ctl.py up': Sunucuyu acar (60-90 sn).",
        "  * 'python kaggle_ctl.py status': 30 saatlik GPU kotasini canli sorgular.",
        "  * 'python kaggle_ctl.py down': Sunucuyu kapatip kota sayacini durdurur.",
        "- Guvenlik:",
        "  256-bit X-Auth-Token yetkilendirme basligi ile yetkisiz erisimler engellenir."
    ])

    add_placeholder(s11, Inches(6.6), Inches(1.45), Inches(5.9), Inches(5.4),
                    placeholder_title="KAGGLE 2X T4 GPU & CLOUDFLARE TUNEL SEMASI",
                    subtitle="Yerel Arayuz -> Cloudflare Tunel -> Kaggle Daemon -> 2x Tesla T4")


    # =========================================================================
    # SLIDE 12: DENEYSEL SONUÇLAR VE PERFORMANS METRİKLERİ
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Deneysel Sonuclar ve Performans Metrikleri (N=8,486 Test Hucresi)", slide_num=12)

    add_stat_box(s12, Inches(0.8), Inches(1.45), Inches(2.7), Inches(1.2), "%98.11", "Birlesik Ortalama Dogruluk", C_TEAL)
    add_stat_box(s12, Inches(3.8), Inches(1.45), Inches(2.7), Inches(1.2), "%99.95", "Kanser / Malignite Dogrulugu", C_RED)
    add_stat_box(s12, Inches(6.8), Inches(1.45), Inches(2.7), Inches(1.2), "%98.16", "Kanser Kok Hucresi (CSC)", C_PURPLE)
    add_stat_box(s12, Inches(9.8), Inches(1.45), Inches(2.7), Inches(1.2), "%0.45", "Kalibrasyon Hatasi (ECE)", C_BLUE)

    add_card(s12, Inches(0.8), Inches(2.85), Inches(6.2), Inches(4.0),
             title="Gorev Bazli Test Metrikleri",
             subtitle="Egitimde Hic Gorulmemis 8,486 Test Hucresi")
    add_bullet_points(s12, Inches(1.05), Inches(3.6), Inches(5.7), Inches(3.1), [
        "- 4c. Malign Kanser Tespiti: %99.95 Dogruluk | ROC-AUC: 1.000 | ECE: %0.04",
        "- 4a. Saglik Durumu (Normal vs Hasta): %98.52 Dogruluk | ROC-AUC: 0.998",
        "- 4b. Hastalik Arketipi (4 Sinif): %98.46 Dogruluk | ROC-AUC: 0.998",
        "- 4d-Ek. Kanser Kok Hucresi (CSC): %98.16 Dogruluk | ROC-AUC: 0.998",
        "- 4d. 11 Kanser Varyanti: %97.23 Dogruluk | ROC-AUC: 0.997",
        "- 1. 14 Hucre Tipi: %96.37 Dogruluk | ROC-AUC: 0.999",
        "- Modalite Kiyaslamasi: Tek-Hucre: %97.92 | Doku (Bulk RNA): %99.70"
    ])

    add_placeholder(s12, Inches(7.3), Inches(2.85), Inches(5.2), Inches(4.0),
                    placeholder_title="KARISIKLIK MATRISI (CONFUSION MATRIX) & ROC-AUC",
                    subtitle="faz8_ab_benchmark_and_confusion.png | 11 Varyant Hata Matrisi")


    # =========================================================================
    # SLIDE 13: İNTERAKTİF TEST TEZGAHI (STREAMLIT APP.PY)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Interaktif Karar Destek ve Test Tezgahi (Streamlit Web Dashboard)", slide_num=13)

    add_card(s13, Inches(0.8), Inches(1.45), Inches(5.5), Inches(5.4),
             title="Klinisyen ve Arastirmaci Arayuzu",
             subtitle="app.py & app.bat Uzerinden Canli Deneyler")
    add_bullet_points(s13, Inches(1.05), Inches(2.3), Inches(5.0), Inches(4.3), [
        "1. Rastgele Test Hucresi Ornekleyici:",
        "   8,486 gorulmemis test hucresi icinden tek tikla kategori bazli (CSC+, Kanser, Saglikli vb.) hucre ceker.",
        "2. Canli Pseudo-Bulk Biyopsi Sentezleyici:",
        "   5-80 arasi hucreyi Dirichlet dagilimi ile harmanlayarak yapay heterojen tumor dokusu uretir.",
        "3. Gercek Etiket (Ground-Truth) Karsilastirma Kartlari:",
        "   Model tahmini ile gercek laboratuvar etiketini yan yana dogrular.",
        "4. Toplu Stres Testi (Batch Validation):",
        "   20, 50, 100 veya 200 hucrelik grubu tek seferde test edip anlik basari metrikleri uretir.",
        "5. Inline Base64 Render Motoru:",
        "   Starlette ASGI cakismalarini engelleyen sifir-medya-hatasi HTML-DOM grafik motoru."
    ])

    add_placeholder(s13, Inches(6.6), Inches(1.45), Inches(5.9), Inches(5.4),
                    placeholder_title="STREAMLIT WEB ARAYUZU EKRAN GORUNTUSU",
                    subtitle="http://localhost:8501 | Karar Tezgahi Arayuz Gorseli")


    # =========================================================================
    # SLIDE 14: İŞ PAKETİ DURUMU VE GELECEK ÇALIŞMALAR
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Is Paketi Ilerleme Durumu ve Final Teslim Yol Haritasi", slide_num=14)

    add_card(s14, Inches(0.8), Inches(1.45), Inches(5.7), Inches(5.4),
             title="Tamamlanan Asamalar (Ara Rapor Donemi)",
             subtitle="Faz 1 - Faz 8 Arasi Tum Hedefler Gerceklestirildi")
    add_bullet_points(s14, Inches(1.05), Inches(2.3), Inches(5.2), Inches(4.3), [
        "- IP-1: Veri Tabani ve Coklu-Doku Entegrasyonu (%100 Tamamlandi)",
        "  38,263 ornek, 14 hucre tipi, 11 kanser varyanti standartlastirildi.",
        "- IP-2: scMamba2 Backbone ve BiMamba Egitimi (%100 Tamamlandi)",
        "  State-space mimarisi ve multi-task loss ile %98.11 basariya ulasildi.",
        "- IP-3: XAI ve 4-Panelli Isı Haritasi Motoru (%100 Tamamlandi)",
        "  Input x Gradient atifi ve vektorel poster uretimi dogrulandi.",
        "- IP-4: Kaggle GPU Koprusu ve Streamlit Tezgahi (%100 Tamamlandi)",
        "  Cloudflare tuneli ve interaktif karar tezgahi canliya alindi.",
        "- IP-5: Acik Kaynak GitHub Yedeklemesi (%100 Tamamlandi)",
        "  Tum kodlar, agirliklar ve raporlar GitHub uzerinde arsivlendi."
    ])

    add_card(s14, Inches(6.8), Inches(1.45), Inches(5.7), Inches(5.4),
             title="Gelecek Calismalar (Final Donemi)",
             subtitle="Klinik Etkiyi Artiracak Ek Gelistirmeler")
    add_bullet_points(s14, Inches(7.05), Inches(2.3), Inches(5.2), Inches(4.3), [
        "1. Mekansal Transkriptomik (Spatial Transcriptomics):",
        "   10x Visium / Xenium verileri entegre edilerek tumor ve CSC hucreleri 2D/3D doku koordinatlarinda haritalanacak.",
        "2. Kemoterapi ve Ilac Yaniti Tahmini (Drug Sensitivity):",
        "   GDSC / PRISM veri tabanlariyla model birlestirilerek tespit edilen kanser varyantina en etkili kemoterapotik ajanlar onerilecek.",
        "3. Dis Bagimsiz Kohort Dogrulamasi:",
        "   TCGA ve CELLxGENE uzerinden yeni hasta gruplarinda genellestirilebilirlik testleri genisletilecek.",
        "4. Akademik Bildiri / Makale:",
        "   Elde edilen mimari yenilik ve %98.11'lik basarim uluslararasi bir konferans bildirisine donusturulecek."
    ])


    # =========================================================================
    # SLIDE 15: TEŞEKKÜR VE SORU-CEVAP (Q&A SLIDE)
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s15, C_NAVY_DARK)

    deco = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(7.35), prs.slide_width, Inches(0.15))
    deco.fill.solid()
    deco.fill.fore_color.rgb = C_TEAL
    deco.line.fill.background()

    tb = s15.shapes.add_textbox(Inches(1.0), Inches(1.3), Inches(11.3), Inches(1.0))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    add_run_text(p, "TESEKKUR EDERIM", 38, color=C_TEXT_LIGHT, bold=True)

    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    add_run_text(p2, "Sorulariniz ve Katkilariniz Icin Hazirim", 16, color=C_BLUE)
    p2.space_before = Pt(8)

    q_card = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(2.0), Inches(3.0), Inches(9.33), Inches(3.5))
    q_card.fill.solid()
    q_card.fill.fore_color.rgb = C_NAVY_MED
    q_card.line.color.rgb = RGBColor(51, 65, 85)

    tb_q = s15.shapes.add_textbox(Inches(2.4), Inches(3.3), Inches(8.5), Inches(2.9))
    tfq = tb_q.text_frame
    tfq.word_wrap = True

    p = tfq.paragraphs[0]
    add_run_text(p, "[Proje Basligi]:", 11, color=C_TEXT_MUTED)
    p_sub = tfq.add_paragraph()
    add_run_text(p_sub, "scMamba2 + BiMamba ile Hiyerarsik Klinik Teshis ve Aciklanabilir Yapay Zeka Sistemi", 13, color=C_TEXT_LIGHT, bold=True)
    p_sub.space_after = Pt(12)

    p = tfq.add_paragraph()
    add_run_text(p, "Hazirlayan: Mithat Yasin Celiktas", 12, color=C_TEXT_LIGHT, bold=True)
    p.space_after = Pt(10)

    p = tfq.add_paragraph()
    add_run_text(p, "GitHub Deposu: https://github.com/MithatYasinCeliktas/tasarimProje", 11.5, color=C_BLUE)
    p.space_after = Pt(10)

    p = tfq.add_paragraph()
    add_run_text(p, "Canli Demo: streamlit run app.py (Yerel Test Portu: 8501)", 11.5, color=C_TEAL)

    # Kaydet
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prs.save(output_path)
    print(f"Sunum basariyla olusturuldu: {output_path}")

if __name__ == "__main__":
    out_file = r"scMamba2_BiMamba_Proje_Tasarim_Ara_Raporu_Sunumu.pptx"
    create_deck(out_file)
