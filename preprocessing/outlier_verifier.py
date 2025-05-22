import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os

class OutlierVerifier:
    def __init__(self, master, csv_path, image_root):
        self.master = master
        self.master.title("Outlier Verification Tool")
        
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.index = 0
        self.results = []

        # UI 요소
        self.img_label = tk.Label(master)
        self.img_label.pack()

        self.info_label = tk.Label(master, text="", font=("Arial", 12))
        self.info_label.pack()

        btn_frame = tk.Frame(master)
        btn_frame.pack()

        self.yes_btn = tk.Button(btn_frame, text="이상치 (Y)", command=lambda: self.save_result('y'), width=15, bg='red', fg='white')
        self.yes_btn.pack(side="left", padx=10, pady=10)

        self.no_btn = tk.Button(btn_frame, text="정상 (N)", command=lambda: self.save_result('n'), width=15, bg='green', fg='white')
        self.no_btn.pack(side="right", padx=10, pady=10)

        self.load_image()

        master.bind("<Left>", lambda e: self.prev_image())
        master.bind("<Right>", lambda e: self.next_image())

    def load_image(self):
        if self.index >= len(self.df):
            messagebox.showinfo("완료", "모든 이미지 검수가 완료되었습니다!")
            self.save_csv()
            self.master.quit()
            return

        row = self.df.iloc[self.index]
        img_path = os.path.join(self.image_root, row['image_path'].replace("\\", "/"))

        try:
            img = Image.open(img_path)
            img = img.resize((400, 300))
            self.photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.photo)

            self.info_label.config(text=f"Index: {self.index+1}/{len(self.df)}\n"
                                        f"Class: {row['class']}\n"
                                        f"Path: {row['image_path']}\n"
                                        f"Distance: {row['mean_cosine_distance']:.4f}")
        except Exception as e:
            messagebox.showerror("Error", f"이미지 로드 실패:\n{img_path}\n{e}")
            self.next_image()

    def save_result(self, label):
        row = self.df.iloc[self.index]
        self.results.append({'image_path': row['image_path'], 'class': row['class'], 'label': label})
        self.index += 1
        self.load_image()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            if self.results:
                self.results.pop()
            self.load_image()

    def next_image(self):
        self.index += 1
        self.load_image()

    def save_csv(self):
        result_df = pd.DataFrame(self.results)
        result_df.to_csv("outlier_verification_gui.csv", index=False)
        messagebox.showinfo("저장", "검수 결과가 outlier_verification_gui.csv에 저장되었습니다.")

if __name__ == "__main__":
    root = tk.Tk()
    app = OutlierVerifier(root, "./preprocessing/top500_outliers.csv",'')
    root.mainloop()
