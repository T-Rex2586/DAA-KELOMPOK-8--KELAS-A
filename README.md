# Analisis Rute Terpendek pada Graf Koridor (w = 1)

## Deskripsi

Repositori ini berisi implementasi dan pengujian algoritma **BFS** dan **Dijkstra** untuk mencari rute terpendek pada graf koridor (maze) dengan bobot edge seragam (w = 1). Graf dibangkitkan secara otomatis, kemudian digunakan sebagai bahan uji untuk membandingkan performa kedua algoritma.

Fokus utama proyek ini adalah melihat perbedaan **waktu eksekusi, penggunaan memori, dan jumlah operasi**, sekaligus membuktikan bahwa BFS dan Dijkstra menghasilkan jarak terpendek yang sama pada graf unweighted.

## Informasi Proyek

- Mata Kuliah : Desain dan Analisis Algoritma

- Kelompok:  8

  - Theodosius Rexy Mahardika (L0224025)
  - Farrel Naufal Maghribi (L0224031)
  - Adelia Putri Hapsari (L0224029)

- Topik : Shortest Path pada Graf Koridor

- Pendekatan : Implementasi dan benchmarking empiris

## Algoritma

Program membandingkan dua algoritma berikut:

1. **Dijkstra (Baseline)**\
   Digunakan sebagai pembanding umum untuk shortest path. Pada program ini Dijkstra diimplementasikan menggunakan priority queue (min-heap).

   - Kompleksitas waktu: \(O((V + E) \log V)\)
   - Cocok untuk graf berbobot non-negatif

2. **BFS (Optimized)**\
   Digunakan untuk graf dengan bobot edge seragam (w = 1).

   - Kompleksitas waktu: \(O(V + E)\)
   - Lebih efisien untuk graf unweighted

## Dataset dan Graf

- Graf berbentuk grid/koridor (maze)
- Dibangkitkan menggunakan pendekatan DFS maze generation
- Tidak berarah dan terhubung
- Semua edge memiliki bobot 1

Setiap node direpresentasikan dalam format `x_y`, misalnya `0_0`, `3_5`.

## Struktur Folder

```
DAA_Project/
├── data/                 # File JSON graf hasil generate
├── DAAPROJEK.py          # Program utama
├── README.md             # Dokumentasi
```

## Format Data JSON

Graf disimpan dalam format JSON sederhana yang berisi daftar node, edge, serta source dan target. Contoh singkat:

```json
{
  "nodes": ["0_0", "1_0"],
  "edges": [{"u": "0_0", "v": "1_0", "w": 1}],
  "source": "0_0",
  "target": "1_0"
}
```

## Cara Menjalankan

Jalankan program utama dengan perintah:

```bash
python DAAPROJEK.py
```

Program akan:

- Membuat graf koridor
- Menyimpan graf ke folder `data/`
- Menjalankan BFS dan Dijkstra
- Melakukan benchmark waktu dan memori
- Menampilkan grafik dan tabel hasil

## Hasil dan Analisis

Berdasarkan hasil pengujian:

- BFS dan Dijkstra selalu menghasilkan jarak terpendek yang sama
- BFS berjalan lebih cepat dibandingkan Dijkstra
- Penggunaan memori BFS lebih rendah

Perbedaan performa semakin terlihat saat ukuran graf semakin besar.

## Kesimpulan

Untuk graf dengan bobot edge seragam (w = 1), **BFS adalah pilihan yang lebih efisien** dibandingkan Dijkstra. Dijkstra tetap relevan sebagai algoritma umum untuk graf berbobot, namun pada kasus ini BFS memberikan hasil optimal dengan overhead yang lebih kecil.

Dokumentasi ini dibuat untuk menjelaskan alur program dan hasil eksperimen secara ringkas dan sesuai dengan konteks perkuliahan.

