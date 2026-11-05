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
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_path: str):
    prs = Presentation()
    # 16:9 Geniş Ekran Standardı
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

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
        p0.text = category.upper()
        p0.font.size = Pt(9.5)
        p0.font.bold = True
        p0.font.color.rgb = C_BLUE

        p1 = tf.add_paragraph()
        p1.text = title
        p1.font.size = Pt(19)
        p1.font.bold = True
        p1.font.color.rgb = C_NAVY_DARK

        # Sayfa Numarası & Footer
        if slide_num:
            ftr = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(11.7), Inches(0.35))
            ftf = ftr.text_frame
            ftf.margin_left = ftf.margin_top = ftf.margin_right = ftf.margin_bottom = 0
            fp = ftf.paragraphs[0]
            fp.text = f"Mithat Yasin Çeliktaş | scMamba2 + BiMamba Hiyerarşik Klinik Teşhis & XAI Sistemi                                                                            Slayt {slide_num} / 15"
            fp.font.size = Pt(8.5)
            fp.font.color.rgb = C_TEXT_MUTED

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
            p.text = title
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = C_NAVY_DARK

            if subtitle:
                p2 = tf.add_paragraph()
                p2.text = subtitle
                p2.font.size = Pt(9.5)
                p2.font.color.rgb = C_TEXT_MUTED
        return card

    def add_placeholder(slide, left, top, width, height, placeholder_title, subtitle="Isı Haritası / Grafik Görseli"):
        box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
        box.fill.solid()
        box.fill.fore_color.rgb = C_PH_BG
        box.line.color.rgb = C_PH_BORDER
        box.line.width = Pt(1.5)

        tb = slide.shapes.add_textbox(left + Inches(0.2), top + height/2 - Inches(0.7), width - Inches(0.4), Inches(1.4))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p1 = tf.paragraphs[0]
        p1.text = "🖼️ [ GÖRSEL / ISI HARİTASI ALANI ]"
        p1.alignment = PP_ALIGN.CENTER
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = C_BLUE

        p2 = tf.add_paragraph()
        p2.text = placeholder_title
        p2.alignment = PP_ALIGN.CENTER
        p2.font.size = Pt(11)
        p2.font.bold = True
        p2.font.color.rgb = C_NAVY_DARK

        p3 = tf.add_paragraph()
        p3.text = f"({subtitle})"
        p3.alignment = PP_ALIGN.CENTER
        p3.font.size = Pt(9)
        p3.font.italic = True
        p3.font.color.rgb = C_TEXT_MUTED
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
        p.text = stat_value
        p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = accent_color

        p2 = tf.add_paragraph()
        p2.text = stat_label
        p2.alignment = PP_ALIGN.CENTER
        p2.font.size = Pt(9.5)
        p2.font.bold = True
        p2.font.color.rgb = C_NAVY_DARK

    def add_bullet_points(slide, left, top, width, height, points):
        tb = slide.shapes.add_textbox(left, top, width, height)
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        for i, pt_text in enumerate(points):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = pt_text
            p.font.size = Pt(10.5)
            p.font.color.rgb = C_NAVY_DARK
            p.space_after = Pt(8)
            p.level = 0

    # =========================================================================
    # SLIDE 1: BAŞLIK / KAPAK SLAYTI (Dark Premium Theme)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1, C_NAVY_DARK)

    # Arka plan dekoratif geometrik çizgi
    deco = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(7.35), prs.slide_width, Inches(0.15))
    deco.fill.solid()
    deco.fill.fore_color.rgb = C_BLUE
    deco.line.fill.background()

    # Üst Rozet
    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.1), Inches(11.3), Inches(0.5))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ  |  TASARIM VE BİTİRME PROJESİ — ARA RAPOR SUNUMU"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = C_BLUE

    # Ana Başlık
    tb_title = s1.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(11.3), Inches(2.2))
    tft = tb_title.text_frame
    tft.word_wrap = True
    pt = tft.paragraphs[0]
    pt.text = "scMamba2 + BiMamba ile Hiyerarşik Klinik Teşhis ve Açıklanabilir Yapay Zeka (XAI) Karar Destek Sistemi"
    pt.font.size = Pt(28)
    pt.font.bold = True
    pt.font.color.rgb = C_TEXT_LIGHT

    pt_sub = tft.add_paragraph()
    pt_sub.text = "Tek Hücreli (scRNA-seq) ve Toplu Doku (Bulk RNA-seq) Seviyesinde Kanser ve Kanser Kök Hücresi (CSC) Tespiti"
    pt_sub.font.size = Pt(14)
    pt_sub.font.color.rgb = RGBColor(148, 163, 184)
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
    p.text = "👤 Proje Yürütücüsü: Mithat Yasin Çeliktaş"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_LIGHT

    p2 = tfm.add_paragraph()
    p2.text = "🎯 Danışman / Ders: Tasarım ve Bitirme Projesi Ara Rapor Değerlendirme Jürisi"
    p2.font.size = Pt(11.5)
    p2.font.color.rgb = RGBColor(203, 213, 225)
    p2.space_before = Pt(6)

    p3 = tfm.add_paragraph()
    p3.text = "🔗 Açık Kaynak GitHub Deposu: https://github.com/MithatYasinCeliktas/tasarimProje"
    p3.font.size = Pt(10.5)
    p3.font.color.rgb = C_BLUE
    p3.space_before = Pt(6)

    p4 = tfm.add_paragraph()
    p4.text = "📊 Canlı Ar-Ge Test Tezgahı: Streamlit Web Platformu (Yerel Port: 8501) | Kaggle 2x Tesla T4 GPU Hibrit Köprüsü"
    p4.font.size = Pt(10.5)
    p4.font.color.rgb = C_TEAL
    p4.space_before = Pt(4)


    # =========================================================================
    # SLIDE 2: KLİNİK MOTİVASYON VE PROBLEM TANIMI
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Klinik Problem Tanımı ve Biyolojik Motivasyon", slide_num=2)

    # Sol Kart: Biyolojik Zorluk
    add_card(s2, Inches(0.8), Inches(1.45), Inches(5.6), Inches(4.2),
             title="🧬 Biyolojik Problem: Tümör Heterojenitesi & CSC",
             subtitle="Geleneksel Biyopsilerin ve Doku Ortalamasının Yetersizliği")
    add_bullet_points(s2, Inches(1.05), Inches(2.25), Inches(5.1), Inches(3.2), [
        "• Bulk RNA-seq Maskelemesi: Klasik doku biyopsileri milyonlarca hücrenin ortalamasını alır; nadir ancak en ölümcül alt popülasyonları gizler.",
        "• Kanser Kök Hücreleri (CSC): Kemoterapi ve radyoterapiye dirençli, nüksetme ve metastazı başlatan asıl kök hücre grubudur.",
        "• Klinik İhtiyaç: Tek tek hücre düzeyinde ve doku karışımlarında kök hücre sinyallerini (CD44, ALDH1A1, PROM1) ayırt edebilecek hassas bir tanı motoru şarttır."
    ])

    # Sağ Kart: Hesaplamalı Bilişim Zorluğu
    add_card(s2, Inches(6.8), Inches(1.45), Inches(5.7), Inches(4.2),
             title="💻 Bilişimsel Zorluk: 60,000+ Gen ve Dropout Gürültüsü",
             subtitle="Neden Transformer Mimarileri Bu Ölçekte Tıkanıyor?")
    add_bullet_points(s2, Inches(7.05), Inches(2.25), Inches(5.2), Inches(3.2), [
        "• Aşırı Yüksek Boyutluluk: İnsan genomunda 60,700 gen bulunur. Hücre başına seyreklik (sparsity/dropout) %85-90 düzeyindedir.",
        "• Transformer Darboğazı O(N²): Self-Attention mekanizması 512 genin ötesinde bellek tüketimini katlar; GPU'larda OOM (Out-of-Memory) oluşturur.",
        "• Çözüm: Lineer O(N) karmaşıklıkla çalışan State-Space Model (SSM) Mamba mimarisi ve çift yönlü BiMamba etkileşim katmanı."
    ])

    # Alt Özet Bandı
    add_card(s2, Inches(0.8), Inches(5.85), Inches(11.7), Inches(1.05), bg_color=C_NAVY_DARK, border_color=C_BLUE)
    tb = s2.shapes.add_textbox(Inches(1.0), Inches(5.95), Inches(11.3), Inches(0.85))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🎯 Temel Hedef: scMamba2 + BiMamba mimarisi ile hem tek hücrede hem de toplu dokuda 6 kademeli hiyerarşik klinik teşhis koymak ve kararları gen seviyesinde açıklanabilir kılmaktır."
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_LIGHT


    # =========================================================================
    # SLIDE 3: PROJENİN AMACI VE ÖZGÜN DEĞERİ
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Projenin Amacı, Kapsamı ve Özgün Değeri (Novelty)", slide_num=3)

    # 3 Sütunlu İnovasyon Kartları
    add_card(s3, Inches(0.8), Inches(1.5), Inches(3.7), Inches(5.2),
             title="1. scMamba2 + BiMamba Hibrit Mimarisi",
             subtitle="Foundation Model + Çift Yönlü SSM")
    add_bullet_points(s3, Inches(1.0), Inches(2.35), Inches(3.3), Inches(4.1), [
        "• 60,700 Gen Sözlüğü: scMamba2 ön eğitimli temel omurga ile hücrenin global temsili çıkarılır.",
        "• Çift Yönlü BiMamba: Genomik dizilimdeki ileri (5'→3') ve geri (3'→5') gen düzenleme etkileşimleri modellenir.",
        "• Lineer O(N) Verimlilik: Transformer'lara kıyasla 4x daha hızlı çıkarım ve düşük bellek tüketimi."
    ])

    add_card(s3, Inches(4.8), Inches(1.5), Inches(3.7), Inches(5.2),
             title="2. 6 Kademeli Hiyerarşik Teşhis",
             subtitle="Tek Aşamalı Sınıflandırıcılar Yerine Ağaç Yapısı")
    add_bullet_points(s3, Inches(5.0), Inches(2.35), Inches(3.3), Inches(4.1), [
        "• Düz multiclass yerine onkolojik karar ağacı simüle edilir.",
        "• Hücre Tipi (14 Sınıf)",
        "• Sağlık Durumu (Normal vs Hasta)",
        "• Hastalık Arketipi (COVID, MS, Kanser)",
        "• Malignite / Kanser Tespiti",
        "• 11 Kanser Varyantı (Organ Kökeni)",
        "• Kanser Kök Hücresi (CSC) Varlığı"
    ])

    add_card(s3, Inches(8.8), Inches(1.5), Inches(3.7), Inches(5.2),
             title="3. Açıklanabilir XAI & Isı Haritaları",
             subtitle="Kara Kutu Yapay Zekayı Kırma")
    add_bullet_points(s3, Inches(9.0), Inches(2.35), Inches(3.3), Inches(4.1), [
        "• Input × Gradient Atıfı: Kararı belirleyen Top-18 kilit gen hekime gösterilir.",
        "• 16 Biyolojik Yolak Skoru: Kök hücre, EMT metastaz ve hipoksi aktivasyonu ölçülür.",
        "• 4-Panelli Master Heatmap: Güven, gen, varyant ve yolak seviyesinde tek parça tanı raporu."
    ])


    # =========================================================================
    # SLIDE 4: SİSTEM MİMARİSİ VE AKIŞ ŞEMASI
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Sistem Mimarisi ve Uçtan Uca Veri Akışı", slide_num=4)

    # Sol Kolon: Mimari Bileşenleri
    add_card(s4, Inches(0.8), Inches(1.45), Inches(5.3), Inches(5.4),
             title="⚙️ Mimari Bileşenleri ve Tensör Boyutları",
             subtitle="scMamba2 + BiMamba Entegrasyonu")
    add_bullet_points(s4, Inches(1.05), Inches(2.25), Inches(4.8), Inches(4.4), [
        "1. Girdi Katmanı (AnnData / Tensör):",
        "   • tokens_genes (B, 1024) & tokens_vals (B, 1024)",
        "   • panel_expr (B, 512): Kromozom sıralı Master Genler",
        "   • bio_sigs_norm (B, 16): Biyolojik Yolak Z-skorları",
        "2. scMamba2 Backbone (Pretrained 512-d Temsil):",
        "   • ContinuousValueEncoder + State-Space Blokları",
        "3. Gated BiMamba Karar Motoru (2 Katmanlı):",
        "   • İleri SSM + Geri SSM + Kapı Füzyon Ağı",
        "4. Çok Görevli Kayıp Fonksiyonu (Multi-Task Loss):",
        "   • Focal Loss (Dengesiz Kanser Varyantları)",
        "   • CSC Weighted BCE & Cosine Prototip Kaybı",
        "   • ECE Kalibrasyon Cezalandırması"
    ])

    # Sağ Kolon: Mimari Şema İçin Ayrılmış Alan
    add_placeholder(s4, Inches(6.4), Inches(1.45), Inches(6.1), Inches(5.4),
                    placeholder_title="SİSTEM VE MODEL MİMARİSİ AKIŞ ŞEMASI",
                    subtitle="scMamba2 Backbone -> Gated BiMamba Head -> 6 Teşhis Başlığı")


    # =========================================================================
    # SLIDE 5: 6 KADEMELİ HİYERARŞİK KLİNİK TEŞHİS AĞACI
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "6 Kademeli Hiyerarşik Klinik Karar Ağacı", slide_num=5)

    # 6 Seviyeyi İki Sütun Halinde Kartlarla Göster
    levels = [
        ("Seviye 1: Hücre Tipi (Cell Type)", "14 Temel Biyolojik Sınıf",
         "Malign Kanser Hücresi, T Lenfosit, B Lenfosit, Makrofaj/Monosit, Epitelyal Hücre, Nöron, Astrosit, Oligodendrosit, Fibroblast, Endotel vb.", C_BLUE),
        ("Seviye 4a: Genel Sağlık Durumu", "2 Sınıf: Sağlıklı vs. Hastalıklı",
         "Örneğin sağlıklı bir donörden mi yoksa patolojik inflamasyon/tümör taşıyan bir hastadan mı geldiğini doğrular.", C_TEAL),
        ("Seviye 4b: Hastalık Arketipi", "4 Ana Klinik Kategori",
         "1. Sağlıklı Kontrol  |  2. Akut Viral Enfeksiyon (COVID-19)  |  3. Otoimmün Nöro-İnflamasyon (MS)  |  4. Kanser / Malignite", C_PURPLE),
        ("Seviye 4c: Malignite / Kanser Tespiti", "2 Sınıf: Kanser Değil vs. Malign Kanser",
         "Hastalığın kesin onkolojik tümör hücresi olup olmadığını %99.95 doğrulukla belirler.", C_RED),
        ("Seviye 4d: 11 Kanser Varyantı (Organ Kökeni)", "11 Pan-Kanser Moleküler Varyantı",
         "Non-Cancer, Akciğer (LUAD), A549, Kronik Miyeloid Lösemi (CML), Böbrek (RCC), Rahim (UCEC), Özofagus (ESCA), Miyelom (MYE), Tiroid (THCA), Yumurtalık (OV), Pankreas (PAAD).", C_BLUE),
        ("Seviye 4d-Ek: Kanser Kök Hücresi (CSC)", "2 Sınıf: CSC- Negatif vs. CSC+ POZİTİF",
         "CD44, ALDH1A1 ve CytoTRACE entropisi ile nüks ve kemoterapi direnci potansiyeli yüksek kök hücre karakterini (%98.16 doğruluk) saptar.", C_RED),
    ]

    for idx, (l_title, l_sub, l_desc, l_col) in enumerate(levels):
        r = idx // 2
        c = idx % 2
        l_x = Inches(0.8) + c * Inches(5.95)
        l_y = Inches(1.45) + r * Inches(1.75)

        add_card(s5, l_x, l_y, Inches(5.75), Inches(1.6), title=l_title, subtitle=l_sub)
        # Yan çizgi
        bar = s5.shapes.add_shape(MSO_SHAPE.RECTANGLE, l_x, l_y, Inches(0.08), Inches(1.6))
        bar.fill.solid()
        bar.fill.fore_color.rgb = l_col
        bar.line.fill.background()

        tb = s5.shapes.add_textbox(l_x + Inches(0.2), l_y + Inches(0.75), Inches(5.35), Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = l_desc
        p.font.size = Pt(9.5)
        p.font.color.rgb = C_NAVY_DARK


    # =========================================================================
    # SLIDE 6: VERİ SETİ VE ÖZELLİK MÜHENDİSLİĞİ
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Çok Dokulu Birleşik Veri Seti ve Özellik Mühendisliği", slide_num=6)

    # İstatistik Kutuları
    add_stat_box(s6, Inches(0.8), Inches(1.45), Inches(2.7), Inches(1.2), "38,263", "Toplam Hücre & Doku", C_BLUE)
    add_stat_box(s6, Inches(3.8), Inches(1.45), Inches(2.7), Inches(1.2), "8,486", "Görülmemiş Test Hücresi", C_TEAL)
    add_stat_box(s6, Inches(6.8), Inches(1.45), Inches(2.7), Inches(1.2), "512", "Master Tanı Gen Paneli", C_PURPLE)
    add_stat_box(s6, Inches(9.8), Inches(1.45), Inches(2.7), Inches(1.2), "16", "Biyolojik Yolak Skoru", C_RED)

    # Sol Kart: Veri Seti Bölünmesi
    add_card(s6, Inches(0.8), Inches(2.85), Inches(5.7), Inches(4.0),
             title="📊 Katı Veri Bölünmesi (No Data-Leakage)",
             subtitle="Eğitim, Doğrulama ve Dış Test Kümeleri")
    add_bullet_points(s6, Inches(1.05), Inches(3.6), Inches(5.2), Inches(3.1), [
        "• Eğitim Seti (Train): 25,791 Örnek (%67.4)",
        "• Doğrulama Seti (Validation): 3,986 Örnek (%10.4) — Erken durdurma ve hiperparametre optimizasyonu.",
        "• Dış Test Seti (Unseen Test): 8,486 Örnek (%22.2) — Modelin eğitimde hiç görmediği hücre ve hastalar.",
        "• Modalite Dağılımı: 33,943 tekil hücre (scRNA-seq) + 4,320 Dirichlet ağırlıklı sentetik biyopsi (Bulk RNA-seq)."
    ])

    # Sağ Kart: 16 Biyolojik Yolak
    add_card(s6, Inches(6.8), Inches(2.85), Inches(5.7), Inches(4.0),
             title="🧪 16 Biyolojik Yolak İmzası (Feature Engineering)",
             subtitle="Moleküler Sapmaları Özetleyen Z-Skorları")
    add_bullet_points(s6, Inches(7.05), Inches(3.6), Inches(5.2), Inches(3.1), [
        "• Kanser ve Kök Hücre: CSC Stemness, Proliferasyon (MKI67/TOP2A), EMT Metastaz (VIM/SNAI1), Hipoksi (HIF1A/LDHA).",
        "• İmmün & İnflamasyon: Viral Yanıt (ISG15/IFIT1), MS Nöro-İnflamasyon (GFAP/APOE), Makrofaj SPP1 & C1QC İmzaları, T-Hücre Tükenmişliği.",
        "• Soy ve Kararsızlık: Epitelyal, Nöral/Glial, Hematopoietik soylar, CytoTRACE Kök Hücre Entropisi ve Kromozomal CNV Kararsızlığı."
    ])


    # =========================================================================
    # SLIDE 7: AÇIKLANABİLİR YAPAY ZEKA (XAI) VE INPUT X GRADIENT
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Açıklanabilir Yapay Zeka (XAI) ve Karar Gerekçelendirme", slide_num=7)

    # Sol Kart: XAI Metodolojisi
    add_card(s7, Inches(0.8), Inches(1.45), Inches(5.5), Inches(5.4),
             title="🔍 Kara Kutu Engeli Nasıl Aşıldı?",
             subtitle="Klinik Doğrulama İçin Matematiksel Atıf")
    add_bullet_points(s7, Inches(1.05), Inches(2.3), Inches(5.0), Inches(4.3), [
        "1. BiMamba Input × Gradient Atıfı:",
        "   Modelin karar çıktılarından geriye doğru gradyan akıtılır. Her genin gradyan büyüklüğü ile ifade seviyesi çarpılarak salience (önem) skoru hesaplanır.",
        "   Importance(g) = |x · ∂Loss/∂x| + 0.35 · (x / x_max)",
        "2. Top-18 Biyobelirteç Gen Seçimi:",
        "   Karara en yüksek etkiyi yapan ilk 18 gen anlık olarak sıralanır ve sağlıklı kontrol referansı ile karşılaştırılır.",
        "3. Kanser Prototip Benzerlik Matrisi:",
        "   Gizli uzaydaki 11 kanser prototipi ile hastanın hücresi arasındaki Kosinüs Benzerliği [0, 100] aralığında ölçülür.",
        "4. Biyolojik Anotasyon Sözlüğü:",
        "   Model, her genin klinik fonksiyonunu (ör. CD44: Kök Hücre, MPO: Miyeloid Lösemi) hekimin anlayacağı dilde raporlar."
    ])

    # Sağ Kart: XAI Şeması / Isı Haritası Alanı
    add_placeholder(s7, Inches(6.6), Inches(1.45), Inches(5.9), Inches(5.4),
                    placeholder_title="XAI KARAR AĞIRLIKLARI VE PROTOTİP UYUM GRAFİĞİ",
                    subtitle="Input x Gradient Gen Sıralaması & Kosinüs Benzerlik Çubukları")


    # =========================================================================
    # SLIDE 8: 4-PANELLİ DİNAMİK TEŞHİS RAPORU (MASTER POSTER)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "4-Panelli Dinamik Klinik Teşhis Raporu (Genel Bakış)", slide_num=8)

    # Sol Kolon: Panellerin Açıklaması
    add_card(s8, Inches(0.8), Inches(1.45), Inches(4.5), Inches(5.4),
             title="📋 Master Raporun 4 Katmanı",
             subtitle="3000x2025 px Yüksek Çözünürlüklü Vektörel Çıktı")
    add_bullet_points(s8, Inches(1.0), Inches(2.3), Inches(4.1), Inches(4.3), [
        "• Panel A (Sol-Üst): Hiyerarşik Klinik Karar Ağacı & Kalibre Güven Yüzdeleri (%95 klinik hedef eşiği).",
        "• Panel B (Sağ-Üst): Kararı Belirleyen Top-18 Biyobelirteç Geni ve XAI Isı Haritası (Magma renk paleti).",
        "• Panel C (Sol-Alt): 11 Kanser Varyantı Prototip Örüntü Uyum Isı Haritası (YlOrRd renk paleti).",
        "• Panel D (Sağ-Alt): 16 Biyolojik Yolak, Kök Hücre (CSC) ve Kromozomal CNV Isı Haritası (RdBu_r paleti).",
        "• İndirilebilir Format: Tek tıkla HD PNG klinik arşiv çıktısı."
    ])

    # Sağ Kolon: [BÜYÜK ALAN] 4-Panelli Master Poster Alanı
    add_placeholder(s8, Inches(5.6), Inches(1.45), Inches(6.9), Inches(5.4),
                    placeholder_title="4-PANELLİ BİRLEŞİK MASTER TEŞHİS POSTERİ (PANEL A + B + C + D)",
                    subtitle="latest_diagnostic_report.png | 3000x2025 px HD Grafik Alanı")


    # =========================================================================
    # SLIDE 9: PANEL A VE PANEL B DETAYLI İNCELEMESİ
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Panel A (Teşhis / Güven) & Panel B (Top-18 Gen Heatmap)", slide_num=9)

    # Sol Bölüm: Panel A
    add_card(s9, Inches(0.8), Inches(1.45), Inches(5.7), Inches(2.2),
             title="Panel A: Hiyerarşik Teşhis ve Kalibre Güven Çubukları",
             subtitle="Modelin 6 Kademedeki Karar Kesinliği")
    add_bullet_points(s9, Inches(1.0), Inches(2.25), Inches(5.3), Inches(1.3), [
        "• 6 aşamanın tamamı için tahmin edilen sınıf, gerçek etiket ve güven yüzdesi.",
        "• %95 güven eşiği referans çizgisi. Yeşil çubuklar doğru, kırmızı çubuklar sapmayı gösterir."
    ])
    add_placeholder(s9, Inches(0.8), Inches(3.8), Inches(5.7), Inches(3.05),
                    placeholder_title="PANEL A: HİYERARŞİK GÜVEN ÇUBUKLARI GRAFİĞİ",
                    subtitle="fig_panel_a | Model Güven Dağılımı")

    # Sağ Bölüm: Panel B
    add_card(s9, Inches(6.8), Inches(1.45), Inches(5.7), Inches(2.2),
             title="Panel B: Top-18 Biyobelirteç Gen Feature Heatmap'i",
             subtitle="Hasta İfadesi vs. Sağlıklı ve Kanser Referansları")
    add_bullet_points(s9, Inches(7.0), Inches(2.25), Inches(5.3), Inches(1.3), [
        "• Karara en çok etki eden ilk 18 gen (CD44, ALDH1A1, EPCAM, MKI67 vb.).",
        "• 4 Satırlı Isı Haritası: Hasta İfadesi | Eşleşen Ref | Sağlıklı Ref | BiMamba XAI Ağırlığı."
    ])
    add_placeholder(s9, Inches(6.8), Inches(3.8), Inches(5.7), Inches(3.05),
                    placeholder_title="PANEL B: TOP-18 GEN FEATURE & XAI ISI HARİTASI",
                    subtitle="fig_panel_b | Magma Colormap Heatmap")


    # =========================================================================
    # SLIDE 10: PANEL C VE PANEL D DETAYLI İNCELEMESİ
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Panel C (Varyant Benzerliği) & Panel D (16 Biyolojik Yolak)", slide_num=10)

    # Sol Bölüm: Panel C
    add_card(s10, Inches(0.8), Inches(1.45), Inches(5.7), Inches(2.2),
             title="Panel C: 11 Kanser Varyantı Prototip Örüntü Uyumu",
             subtitle="Hangi Kansere Ne Derece Benziyor?")
    add_bullet_points(s10, Inches(1.0), Inches(2.25), Inches(5.3), Inches(1.3), [
        "• Hastanın hücre profilinin 11 kanser tipi ile kosinüs benzerliği matrisi.",
        "• 2 Satır: Model Tahmin Olasılığı (%) ve Prototip Örüntü Benzerliği (%).",
        "• Metastatik ve primer ayrımında organ kökeni doğrulaması sağlar."
    ])
    add_placeholder(s10, Inches(0.8), Inches(3.8), Inches(5.7), Inches(3.05),
                    placeholder_title="PANEL C: 11 KANSER VARYANTI ÖRÜNTÜ UYUM ISI HARİTASI",
                    subtitle="fig_panel_c | YlOrRd Colormap Heatmap")

    # Sağ Bölüm: Panel D
    add_card(s10, Inches(6.8), Inches(1.45), Inches(5.7), Inches(2.2),
             title="Panel D: 16 Biyolojik Yolak & Kök Hücre (CSC) İmzaları",
             subtitle="Fonksiyonel Hücresel Durum Sapmaları")
    add_bullet_points(s10, Inches(7.0), Inches(2.25), Inches(5.3), Inches(1.3), [
        "• 16 yolağın Z-skor sapması (+1.5 Z üzeri patolojik yüksek, -1.5 Z altı baskılanmış).",
        "• Kök Hücre (CSC), Proliferasyon, EMT Metastaz ve Viral ISG aktivasyon seviyeleri.",
        "• İkinci satırda BiMamba'nın o yolağa verdiği karar önemi gösterilir."
    ])
    add_placeholder(s10, Inches(6.8), Inches(3.8), Inches(5.7), Inches(3.05),
                    placeholder_title="PANEL D: 16 YOLAK & CSC İMZA ISI HARİTASI",
                    subtitle="fig_panel_d | RdBu_r Diverging Colormap Heatmap")


    # =========================================================================
    # SLIDE 11: BULUT ALTYAPISI - KAGGLE 2X T4 GPU KÖPRÜSÜ
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Hibrit Bulut Altyapısı: Kaggle 2x Tesla T4 GPU Köprüsü", slide_num=11)

    add_card(s11, Inches(0.8), Inches(1.45), Inches(5.5), Inches(5.4),
             title="☁️ Sıfır Maliyetli Hibrit Hesaplama Altyapısı",
             subtitle="Yerel Masaüstü ile Kaggle 2x T4 Entegrasyonu")
    add_bullet_points(s11, Inches(1.05), Inches(2.3), Inches(5.0), Inches(4.3), [
        "• Kaggle GPU Daemon (2x Tesla T4, 32 GB VRAM):",
        "   Arka planda çalışan Python HTTP sunucusu ve Cloudflare tüneli otomatik olarak ayağa kalkar.",
        "• Cloudflare HTTPS Tüneli:",
        "   Dinamik HTTPS tüneli ntfy.sh bildirim protokolüyle yerel bilgisayara bağlanır.",
        "• Sıfır Kota İsrafı (Zero Quota Waste):",
        "   - 'python kaggle_ctl.py up': Sunucuyu açar (60-90 sn).",
        "   - 'python kaggle_ctl.py status': 30 saatlik GPU kotasını canlı sorgular.",
        "   - 'python kaggle_ctl.py down': Sunucuyu kapatıp kota sayacını durdurur.",
        "• Güvenlik:",
        "   256-bit X-Auth-Token yetkilendirme başlığı ile yetkisiz erişimler engellenir."
    ])

    add_placeholder(s11, Inches(6.6), Inches(1.45), Inches(5.9), Inches(5.4),
                    placeholder_title="KAGGLE 2X T4 GPU & CLOUDFLARE TÜNEL ŞEMASI",
                    subtitle="Yerel Arayüz -> Cloudflare Tünel -> Kaggle Daemon -> 2x Tesla T4")


    # =========================================================================
    # SLIDE 12: DENEYSEL SONUÇLAR VE PERFORMANS METRİKLERİ
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Deneysel Sonuçlar ve Performans Metrikleri (N=8,486 Test Hücresi)", slide_num=12)

    # 4 Büyük Metrik Kartı
    add_stat_box(s12, Inches(0.8), Inches(1.45), Inches(2.7), Inches(1.2), "%98.11", "Birleşik Ortalama Doğruluk", C_TEAL)
    add_stat_box(s12, Inches(3.8), Inches(1.45), Inches(2.7), Inches(1.2), "%99.95", "Kanser / Malignite Doğruluğu", C_RED)
    add_stat_box(s12, Inches(6.8), Inches(1.45), Inches(2.7), Inches(1.2), "%98.16", "Kanser Kök Hücresi (CSC)", C_PURPLE)
    add_stat_box(s12, Inches(9.8), Inches(1.45), Inches(2.7), Inches(1.2), "%0.45", "Kalibrasyon Hatası (ECE)", C_BLUE)

    # Sol Kart: Görev Bazlı Detay Tablosu
    add_card(s12, Inches(0.8), Inches(2.85), Inches(6.2), Inches(4.0),
             title="📈 Görev Bazlı Test Metrikleri",
             subtitle="Eğitimde Hiç Görülmemiş 8,486 Test Hücresi")
    add_bullet_points(s12, Inches(1.05), Inches(3.6), Inches(5.7), Inches(3.1), [
        "• 4c. Malign Kanser Tespiti: %99.95 Doğruluk | ROC-AUC: 1.000 | ECE: %0.04",
        "• 4a. Sağlık Durumu (Normal vs Hasta): %98.52 Doğruluk | ROC-AUC: 0.998",
        "• 4b. Hastalık Arketipi (4 Sınıf): %98.46 Doğruluk | ROC-AUC: 0.998",
        "• 4d-Ek. Kanser Kök Hücresi (CSC): %98.16 Doğruluk | ROC-AUC: 0.998",
        "• 4d. 11 Kanser Varyantı: %97.23 Doğruluk | ROC-AUC: 0.997",
        "• 1. 14 Hücre Tipi: %96.37 Doğruluk | ROC-AUC: 0.999",
        "• Modalite Kıyaslaması: Tek-Hücre: %97.92  |  Doku (Bulk RNA): %99.70"
    ])

    # Sağ Alan: Karışıklık Matrisi Placeholder
    add_placeholder(s12, Inches(7.3), Inches(2.85), Inches(5.2), Inches(4.0),
                    placeholder_title="KARIŞIKLIK MATRİSİ (CONFUSION MATRIX) & ROC-AUC",
                    subtitle="faz8_ab_benchmark_and_confusion.png | 11 Varyant Hata Matrisi")


    # =========================================================================
    # SLIDE 13: İNTERAKTİF TEST TEZGAHI (STREAMLIT APP.PY)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "İnteraktif Karar Destek ve Test Tezgahı (Streamlit Web Dashboard)", slide_num=13)

    add_card(s13, Inches(0.8), Inches(1.45), Inches(5.5), Inches(5.4),
             title="🖥️ Klinisyen ve Araştırmacı Arayüzü",
             subtitle="app.py & app.bat Üzerinden Canlı Deneyler")
    add_bullet_points(s13, Inches(1.05), Inches(2.3), Inches(5.0), Inches(4.3), [
        "1. Rastgele Test Hücresi Örnekleyici:",
        "   8,486 görülmemiş test hücresi içinden tek tıkla kategori bazlı (CSC+, Kanser, Sağlıklı vb.) hücre çeker.",
        "2. Canlı Pseudo-Bulk Biyopsi Sentezleyici:",
        "   5-80 arası hücreyi Dirichlet dağılımı ile harmanlayarak yapay heterojen tümör dokusu üretir.",
        "3. Gerçek Etiket (Ground-Truth) Karşılaştırma Kartları:",
        "   Model tahmini ile gerçek laboratuvar etiketini yan yana yeşil/kırmızı rozetlerle doğrular.",
        "4. Toplu Stres Testi (Batch Validation):",
        "   20, 50, 100 veya 200 hücrelik grubu tek seferde test edip anlık başarı metrikleri üretir.",
        "5. Inline Base64 Render Motoru:",
        "   Starlette ASGI çakışmalarını engelleyen sıfır-medya-hatası HTML-DOM grafik motoru."
    ])

    add_placeholder(s13, Inches(6.6), Inches(1.45), Inches(5.9), Inches(5.4),
                    placeholder_title="STREAMLIT WEB ARAYÜZÜ EKRAN GÖRÜNTÜSÜ",
                    subtitle="http://localhost:8501 | Karar Tezgahı Arayüz Görseli")


    # =========================================================================
    # SLIDE 14: İŞ PAKETİ DURUMU VE GELECEK ÇALIŞMALAR
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "İş Paketi İlerleme Durumu ve Final Teslim Yol Haritası", slide_num=14)

    # Sol Kart: Tamamlanan İş Paketleri (Ara Rapor)
    add_card(s14, Inches(0.8), Inches(1.45), Inches(5.7), Inches(5.4),
             title="✅ Tamamlanan Aşamalar (Ara Rapor Dönemi)",
             subtitle="Faz 1 - Faz 8 Arası Tüm Hedefler Gerçekleştirildi")
    add_bullet_points(s14, Inches(1.05), Inches(2.3), Inches(5.2), Inches(4.3), [
        "• İP-1: Veri Tabanı ve Çoklu-Doku Entegrasyonu (%100 Tamamlandı)",
        "   38,263 örnek, 14 hücre tipi, 11 kanser varyantı standartlaştırıldı.",
        "• İP-2: scMamba2 Backbone ve BiMamba Eğitimi (%100 Tamamlandı)",
        "   State-space mimarisi ve multi-task loss ile %98.11 başarıya ulaşıldı.",
        "• İP-3: XAI ve 4-Panelli Isı Haritası Motoru (%100 Tamamlandı)",
        "   Input × Gradient atıfı ve vektörel poster üretimi doğrulandı.",
        "• İP-4: Kaggle GPU Köprüsü ve Streamlit Tezgahı (%100 Tamamlandı)",
        "   Cloudflare tüneli ve interaktif karar tezgahı canlıya alındı.",
        "• İP-5: Açık Kaynak GitHub Yedeklemesi (%100 Tamamlandı)",
        "   Tüm kodlar, ağırlıklar ve raporlar GitHub üzerinde arşivlendi."
    ])

    # Sağ Kart: Gelecek Çalışmalar (Final Dönemi)
    add_card(s14, Inches(6.8), Inches(1.45), Inches(5.7), Inches(5.4),
             title="🔮 Final Dönemi Hedefleri (Gelecek Çalışmalar)",
             subtitle="Klinik Etkiyi Artıracak Ek Geliştirmeler")
    add_bullet_points(s14, Inches(7.05), Inches(2.3), Inches(5.2), Inches(4.3), [
        "1. Mekansal Transkriptomik (Spatial Transcriptomics):",
        "   10x Visium / Xenium verileri entegre edilerek tümör ve CSC hücreleri 2D/3D doku koordinatlarında haritalanacak.",
        "2. Kemoterapi ve İlaç Yanıtı Tahmini (Drug Sensitivity):",
        "   GDSC / PRISM veri tabanlarıyla model birleştirilerek tespit edilen kanser varyantına en etkili kemoterapötik ajanlar önerilecek.",
        "3. Dış Bağımsız Kohort Doğrulaması:",
        "   TCGA ve CELLxGENE üzerinden yeni hasta gruplarında genelleştirilebilirlik testleri genişletilecek.",
        "4. Akademik Bildiri / Makale:",
        "   Elde edilen mimari yenilik ve %98.11'lik başarım uluslararası bir konferans bildirisine dönüştürülecek."
    ])


    # =========================================================================
    # SLIDE 15: TEŞEKKÜR VE SORU-CEVAP (Q&A SLIDE)
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s15, C_NAVY_DARK)

    # Alt çizgi
    deco = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(7.35), prs.slide_width, Inches(0.15))
    deco.fill.solid()
    deco.fill.fore_color.rgb = C_TEAL
    deco.line.fill.background()

    tb = s15.shapes.add_textbox(Inches(1.0), Inches(1.3), Inches(11.3), Inches(1.0))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "TEŞEKKÜR EDERİM"
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(38)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_LIGHT

    p2 = tf.add_paragraph()
    p2.text = "Sorularınız ve Katkılarınız İçin Hazırım"
    p2.alignment = PP_ALIGN.CENTER
    p2.font.size = Pt(16)
    p2.font.color.rgb = C_BLUE
    p2.space_before = Pt(8)

    # Kart
    q_card = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(2.0), Inches(3.0), Inches(9.33), Inches(3.5))
    q_card.fill.solid()
    q_card.fill.fore_color.rgb = C_NAVY_MED
    q_card.line.color.rgb = RGBColor(51, 65, 85)

    tb_q = s15.shapes.add_textbox(Inches(2.4), Inches(3.3), Inches(8.5), Inches(2.9))
    tfq = tb_q.text_frame
    tfq.word_wrap = True

    p = tfq.paragraphs[0]
    p.text = "📌 Proje Başlığı:"
    p.font.size = Pt(11)
    p.font.color.rgb = C_TEXT_MUTED
    p_sub = tfq.add_paragraph()
    p_sub.text = "scMamba2 + BiMamba ile Hiyerarşik Klinik Teşhis ve Açıklanabilir Yapay Zeka Sistemi"
    p_sub.font.size = Pt(13)
    p_sub.font.bold = True
    p_sub.font.color.rgb = C_TEXT_LIGHT
    p_sub.space_after = Pt(12)

    p = tfq.add_paragraph()
    p.text = "👤 Hazırlayan: Mithat Yasin Çeliktaş"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_LIGHT
    p.space_after = Pt(10)

    p = tfq.add_paragraph()
    p.text = "🔗 GitHub Deposu: https://github.com/MithatYasinCeliktas/tasarimProje"
    p.font.size = Pt(11.5)
    p.font.color.rgb = C_BLUE
    p.space_after = Pt(10)

    p = tfq.add_paragraph()
    p.text = "📊 Canlı Demo: streamlit run app.py (Yerel Test Portu: 8501)"
    p.font.size = Pt(11.5)
    p.font.color.rgb = C_TEAL

    # Kaydet
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prs.save(output_path)
    print(f"Sunum başarıyla oluşturuldu: {output_path}")

if __name__ == "__main__":
    out_file = r"c:\Users\mitha\Desktop\tasarımProje\scMamba2_BiMamba_Proje_Tasarim_Ara_Raporu_Sunumu.pptx"
    create_deck(out_file)
