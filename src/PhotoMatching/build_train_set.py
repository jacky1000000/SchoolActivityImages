import os
import json
import random
import torch
import torch.nn.functional as F
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data as imagebind_data

def generate_train_json(
    train_photo_dir="../../data/train_photos",
    checkpoint_path="../../my_checkpoints/imagebind_huge.pth",
    output_json="../../data/imagebind_train.json",
    confidence_threshold=0.185,
    batch_size=32
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 定義貴校各類活動的多樣化英文描述模板
    # should be array of sentences not array of array of sentences
    activity_templates = [
        "students running on a red athletics track during a school sports day",
        "high school track and field relay race competition in an outdoor stadium",
        "students cheering in the stadium stands and holding medals at a school sports meet",
        # 2. 水運會 / 室內游泳比賽 (Swimming Gala)
        "students swimming in lanes during an indoor school swimming gala competition",
        "swimmers diving off starting blocks at a school swimming pool event",
        "students cheering poolside at a school aquatic sports competition",
        # 3. 球類運動與體育課 (Indoor Ball Games & PE)
        "students playing basketball or badminton inside a school indoor gymnasium",
        "high school volleyball or table tennis match in an indoor sports hall",
        "students practicing football on a school outdoor turf pitch",
        # 4. 科學實驗室與 STEM 活動 (Science Lab & STEM)
        "students conducting chemistry experiments with microscopes and test tubes in a science laboratory",
        "high school STEM robotics competition with students building and testing robots",
        "students in white lab coats performing biology experiments in a modern lab",
        # 5. 電腦室與資訊科技比賽 (Computer Lab & Coding)
        "students coding and working on desktop computers in a school computer laboratory",
        "high school students participating in a programming hackathon or IT workshop",
        "teacher demonstrating software on a screen in a modern computer classroom",
        # 6. 校園音樂會與管弦樂團 (Music Concert & Orchestra)
        "school orchestra playing violins, cellos, and flutes on an auditorium stage",
        "students choir singing together in uniforms during a school music concert",
        "student band performing with guitars and drums at a school talent show",
        # 7. 戲劇表演與舞蹈匯演 (Drama & Dance Performance)
        "students acting in costumes under stage spotlights during a school drama play",
        "high school dance team performing synchronized choreography on a stage",
        "theatrical performance in a school hall with dramatic stage lighting",
        # 8. 畢業典禮與頒獎儀式 (Graduation & Prize Giving)
        "students wearing graduation gowns and caps on stage during a graduation ceremony",
        "school principal shaking hands and presenting awards or trophies to students",
        "graduating class taking a formal group photo on stage in academic regalia",
        # 9. 全校早會與專題講座 (School Assembly & Guest Lecture)
        "teachers and students attending a formal assembly in a large school hall",
        "guest speaker giving a presentation with slides in a school auditorium",
        "students sitting in rows listening to a seminar in a lecture theater",
        # 10. 升旗禮與國民教育儀式 (Flag Raising Ceremony)
        "students and teachers standing in neat rows in a school courtyard during a flag raising ceremony",
        "student flag guards marching and raising flags on a school flagpole",
        "formal morning ceremony in the school playground with students in uniform",
        # 11. 校園開放日與社團攤位 (Open Day & Club Booths)
        "students running colorful game booths and display stalls during a school open day",
        "parents and visitors walking around student club exhibition booths in a school playground",
        "bustling school carnival and charity bazaar with handmade posters and banners",
        # 12. 視覺藝術與設計展覽 (Art & Design Exhibition)
        "students viewing paintings, sculptures, and artworks in a school art exhibition",
        "high school students painting on easels and crafting in a bright art studio",
        "display boards showcasing student photography and graphic design projects",
        # 13. 辯論比賽與學術問答 (Debate & Academic Quiz)
        "students in suits speaking at podiums during an inter-school debate competition",
        "high school quiz bowl team pressing buzzers on stage in front of judges",
        "formal student forum and public speaking contest in a school hall",
        # 14. 圖書館閱讀與小組討論 (Library & Collaborative Learning)
        "students reading books and studying together in a bright modern school library",
        "group discussion around wooden tables in a school learning commons",
        "students browsing bookshelves and researching on laptops in a library",
        # 15. 戶外考察與生態研學 (Field Trip & Outdoor Excursion)
        "students and teachers hiking and observing nature during an outdoor geography field trip",
        "high school class visiting a museum or cultural heritage site together",
        "students taking notes outdoors in a botanical garden or wetland park",
        # 16. 義工服務與社區探訪 (Volunteer & Community Service)
        "student volunteers in matching t-shirts helping elderly or packing charity gift bags",
        "high school students planting trees or cleaning up a beach during a community service day",
        "students smiling and holding volunteer certificates during a charity event",
        # 17. 家政烹飪與文化工作坊 (Cooking & Cultural Workshop)
        "students wearing aprons cooking and baking in a school home economics kitchen",
        "traditional Chinese calligraphy or tea ceremony workshop in a school activity room",
        "students making crafts and festive decorations together in a classroom",
        # 18. 班級大合照與校園生活 (Class Group Photo & Campus Life)
        "smiling high school students and teachers posing for a formal class group photo",
        "students chatting and walking together along a sunny school corridor or courtyard",
        "happy students celebrating together in school uniforms on campus"
    ]


    print(f"⏳ 正在載入 ImageBind 模型至 {device}...")
    model = imagebind_model.imagebind_huge(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)

    # 1. 計算各活動類別的平均文字向量錨點
    category_embeds = []
    with torch.no_grad():
        for templates in activity_templates:
            text_inputs = {ModalityType.TEXT: imagebind_data.load_and_transform_text(templates, device)}
            embeds = F.normalize(model(text_inputs)[ModalityType.TEXT], p=2, dim=-1)
            mean_embed = F.normalize(embeds.mean(dim=0, keepdim=True), p=2, dim=-1)
            category_embeds.append(mean_embed)
    category_matrix = torch.cat(category_embeds, dim=0)

    # 2. 分批掃描 train_photos/ 內的 1,060 張訓練照片
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    train_paths = [
        os.path.join(train_photo_dir, f).replace("\\", "/")
        for f in sorted(os.listdir(train_photo_dir))
        if f.lower().endswith(valid_exts)
    ]

    print(f"📸 正在為 {len(train_paths)} 張訓練集相片進行自動語意標註...")
    all_vision_embeds = []
    with torch.no_grad():
        for i in range(0, len(train_paths), batch_size):
            print(f"Processing batch {i // batch_size + 1} / {(len(train_paths) + batch_size - 1) // batch_size}...")
            batch = train_paths[i:i + batch_size]
            inputs = {ModalityType.VISION: imagebind_data.load_and_transform_vision_data(batch, device)}
            v_embeds = F.normalize(model(inputs)[ModalityType.VISION], p=2, dim=-1)
            all_vision_embeds.append(v_embeds)

    vision_matrix = torch.cat(all_vision_embeds, dim=0)
    similarities = torch.matmul(vision_matrix, category_matrix.T)
    best_scores, best_cat_indices = torch.max(similarities, dim=-1)

    train_items = []
    random.seed(42)
    for idx, img_path in enumerate(train_paths):
        score = best_scores[idx].item()
        if score >= confidence_threshold:
            cat_idx = best_cat_indices[idx].item()
            train_items.append({
                "image_path": img_path,
                "text": activity_templates[cat_idx],
                "confidence": round(score, 4)
            })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(train_items, f, ensure_ascii=False, indent=2)

    print(f"🎉 訓練集 JSON 建立完成！共篩選出 {len(train_items)} 筆高品質圖文配對 -> {output_json}")

if __name__ == "__main__":
    generate_train_json()
