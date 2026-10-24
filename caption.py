import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import torch
from PIL import Image, ImageTk
from transformers import BlipProcessor, BlipForConditionalGeneration

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = None
model = None

def load_ai_model():
    global processor, model
    status_label.config(text="AI Model load ho raha hai, kripya wait karein...")
    select_btn.config(state="disabled")
    generate_btn.config(state="disabled")
    
    # Model load process
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval()
    
    status_label.config(text="Model ready hai! Ab image select karein.")
    select_btn.config(state="normal")

# Image select function
selected_image_path = None

def select_image():
    global selected_image_path
    file_path = filedialog.askopenfilename(
        title="Image choose karein",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp")]
    )
    if file_path:
        selected_image_path = file_path
        
        # Image preview set karein
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_display = ImageTk.PhotoImage(img)
        image_canvas.image = img_display
        image_canvas.config(image=img_display, text="")
        
        caption_box.delete("1.0", tk.END)
        generate_btn.config(state="normal")
        status_label.config(text=f"Selected: {os.path.basename(file_path)}")

# Caption generation function
def run_caption():
    if not selected_image_path:
        messagebox.showwarning("Warning", "Pehle koi image select karein!")
        return

    status_label.config(text="Caption generate ho raha hai...")
    generate_btn.config(state="disabled")
    select_btn.config(state="disabled")
    caption_box.delete("1.0", tk.END)

    def process():
        try:
            image = Image.open(selected_image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    num_beams=5,
                    max_new_tokens=40,
                    repetition_penalty=1.2
                )
            caption = processor.decode(output[0], skip_special_tokens=True)
            
            # UI update
            caption_box.insert(tk.END, caption)
            status_label.config(text="Done!")
        except Exception as e:
            messagebox.showerror("Error", f"Error aaya: {str(e)}")
            status_label.config(text="Error occurred.")
        finally:
            generate_btn.config(state="normal")
            select_btn.config(state="normal")

    threading.Thread(target=process).start()

# --- GUI Window Setup ---
root = tk.Tk()
root.title("AI Image Caption Generator")
root.geometry("450x620")
root.resizable(False, False)
root.configure(bg="#f4f5f7")

# Heading
title_label = tk.Label(root, text="Image Caption Generator", font=("Arial", 16, "bold"), bg="#f4f5f7", fg="#222")
title_label.pack(pady=10)

# Image Preview Box
image_canvas = tk.Label(root, text="Koi image select nahi ki gayi", bg="#e2e5eb", width=38, height=14, relief="groove")
image_canvas.pack(pady=10)

# Select Button
select_btn = tk.Button(root, text="Select Image from Folder", command=select_image, font=("Arial", 10, "bold"), bg="#3b82f6", fg="white", padx=12, pady=6, relief="flat", cursor="hand2")
select_btn.pack(pady=5)

# Generate Button
generate_btn = tk.Button(root, text="Generate Caption", command=run_caption, font=("Arial", 11, "bold"), bg="#10b981", fg="white", padx=15, pady=7, relief="flat", cursor="hand2", state="disabled")
generate_btn.pack(pady=10)

# Status Label
status_label = tk.Label(root, text="Starting app...", font=("Arial", 9, "italic"), bg="#f4f5f7", fg="#555")
status_label.pack(pady=5)

# Output Box
caption_box = tk.Text(root, height=4, width=45, font=("Arial", 10), wrap="word", relief="solid", bd=1)
caption_box.pack(pady=10)

# Background me model load shuru karein
threading.Thread(target=load_ai_model, daemon=True).start()

root.mainloop()
