import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import json
import os
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import warnings
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
warnings.filterwarnings('ignore')
from tqdm import tqdm



# 创建AcsClient实例
client = AcsClient(
    access_key_id,
    access_key_secret,
    "cn-hangzhou"
)


# 检查当机可用字体
def setup_fonts():
    chinese_fonts = ['STSong', 'PingFang SC', 'Heiti SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'WenQuanYi Micro Hei']

    available_font = None
    for font in chinese_fonts:
        if any(font.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            available_font = font
            break

    # 默认
    if not available_font:
        available_font = 'Arial Unicode MS'

    print(f"使用字体: {available_font}")
    return available_font


# 设置停用词表
def get_stopwords():
    stopwords = set([
        "的", "是", "在", "了", "和", "与", "这", "那", "我", "你", "他", "她", "它", "我们", "你们", "他们",
        "这个", "那个", "这些", "那些", "一个", "一些", "有", "没有", "不", "很", "非常", "也", "还", "就",
        "都", "而", "而且", "但是", "但", "如果", "因为", "所以", "可以", "可能", "要", "会", "能", "一", "啊",
        "呢", "吧", "啦", "吗", "么", "哦", "嗯", "哈", "哎", "喔", "啊", "嘛", "呀", "哇", "呵", "哼", "哟",
        "嘿", "哩", "喂", "咦", "哪", "如何", "怎么", "什么", "为什么", "怎样", "多少", "几", "谁", "哪里",
        "何时", "何地", "何人", "怎么办", "多久", "多长时间", "为何", "何", "多", "少", "大", "小",
        "上", "下", "左", "右", "前", "后", "里", "外", "中", "内", "外", "今", "昨", "明", "早", "晚", "年",
        "月", "日", "时", "分", "秒", "天", "周", "星期", "季度", "度", "号", "期", "便", "使", "让", "令",
        "到", "往", "去", "从", "向", "对", "为", "给", "用", "被", "把", "将", "由", "得", "地", "着", "过",
        "了", "来", "去", "进", "出", "起", "落", "开", "关", "当", "如", "若", "全", "半", "多", "少", "等",
        "比", "却", "再", "又", "或", "及", "并", "好", "坏", "低", "高", "和", "与", "约", "连", "这样", "那样",
        "另外", "原来", "其实", "只是", "反而", "不过", "可是", "还是", "尽管", "虽然", "不过", "因此", "总之",
        "之后", "以后", "以前", "之前", "不仅", "不止", "一直", "一向", "仍然", "始终", "总是", "从来", "曾经",
        "最近", "已经", "就要", "刚刚", "立刻", "马上", "顿时", "瞬间", "终于", "整个", "全部", "所有", "每个",
        "各种", "各自", "每次", "继续", "开始", "重新", "随着", "不断", "逐渐", "即使", "哪怕", "如此", "正好",
        "恰好", "真是", "确实", "实在", "反倒", "反而", "越来", "越", "还有", "除了", "另外", "其余", "只有",
        "仅有", "几乎", "差点", "稍微", "略微", "有点", "不太", "微微", "轻轻", "悄悄", "偶尔", "也许", "大概",
        "或许", "可能", "似乎", "好像", "应该", "应当", "必须", "必要", "一定", "肯定", "确定", "相当", "颇",
        "来", "把", "说", "找", "车", "才", "真", "点", "里", "外", "但", "所", "至", "给", "让", "看", "她", "俺",
        "前", "后", "做", "又", "打", "叫", "等", "过", "汽车", "车子", "自己", "这种", "感觉", "觉得", "认为", "想", "本", "样", "种", "类", "次", "回",
        "本来", "他", "她","但是","一点","多","好","了","特别","对于","来说","所以","这个","方面","喜欢","一","现在","不过",
        "下","问题","吧","想","去","还有","如果","个人","因为","真的","如果","的","脸","版",
        "那么","中","也有","刚开始","多久","拿来","你们","多久","点儿","或者","性","作为","带","加上","有一点",
        "不要","咋办","发现","又是","回来","很好","不用","担心","能力","完全","一般","一开","这里","只能","不是",
        "其他","要求","足够","很多","起来","十足","熟悉","不能","十分","车辆","差不多","经常","习惯","毕竟","辆车","同时",
        "太行","只要","经常","一种","整车","高的","一种","看吧","知道","要求","放不下","太少","一会儿","包括",
        "具有","绝对","一样","对比","太多","一代","大家","不是","的话","起来","不能","目前","影响","首先","其次","反正",
        "基本","来讲","来说","看起来","直接","台车","无论是","采用","注意到","好歹","硬要","不得不","遇到","之间","看吧","挑出",
        "车里","车上","后期","不知道","后面","情况下","属于","更加","基本上","基本","当初","小时","一下","出现","等等","原因",
        "来看","高高","符合","以及","回答","开开","十足","然后","不管","当然","不够","车上","无法","敢开","唯一","目前",
        "一个人","心里","几天","过去","路上","顺手","就是","情况下","情况","展现","不到","至少","根本","刚刚好","完全","刚好",
        "建议","电子","体现","展示","接受","没问题","高的","经过","天天","吃不下","很少","不想","搞得","多多少少","我不知道","闻到",
        "到家","亮丽","很重","放大","下雨天","更长","关心","全部","一句话","不论","了解到","有可能","全方位","单独","极强","申请","一名",
        "直线", "双腿", "包围", "熏黑", "养护", "微信", "除非", "天冷", "升降", "讲解", "颇具", "释放", "咨询", "高峰", "较长", "走到", "明确",
        "成人", "宝马", "意义", "几千", "比不上", "看我", "厘米", "至今", "纸巾", "幸好", "稳稳", "胖子", "受不了", "开久", "一项", "背部", "地区",
        "约车", "呼唤", "龙爪", "周到", "玩耍", "认知", "舒畅", "上海", "旧车", "一瞬间", "沉重", "对象", "单纯", "分配", "不合理", "期望", "街道",
        "衣服",  "半路", "地铁", "干脆", "坚持", "怀疑", "第一感觉", "上学", "炭包", "发动", "威武", "跟上", "高点", "出入", "不见",
        "闺蜜", "单一", "电影", "范儿", "接待", "奇怪", "不爽", "杯子", "开锁", "明智", "零食", "代表", "期间", "心理", "高速公路", "好久", "体会",
        "是否", "没事", "加速度", "进步", "眼看", "犹豫", "暖风", "远远", "字母", "喜爱", "所谓", "后段", "主打", "放到", "半个", "检查", "早上",
        "十几", "紧急", "过年", "烦恼", "分布", "打孔", "来了", "奥迪", "算得", "几百", "心意", "深刻", "横向", "小心", "自信", "几个月", "路口",
        "中型", "两周", "例如", "两点", "三元", "取消", "直到", "想法", "据说", "观察", "费劲", "再说", "自行车", "执行", "满载", "大床", "春秋",
        "首选", "一句", "马路", "然而", "到底", "顺利", "减轻", "变化", "会不会", "无需", "想想", "某些", "小区", "看过", "对得起", "关系", "不行",
        "想的","短板","过分","最爱","在乎","两百","同步","打折扣","几点","跑得","详细","专属","消散","契合",
         "用户", "这次", "看上去", "官方", "一台", "到位", "车主", "当时", "级别", "达到", "明显", "保持", "每天",
        "手机", "版本", "较大", "很大", "适应", "主动", "选择", "内部", "更新", "差距", "正常", "厂家", "销售", "主动",
        "上面","力度","合理","第一次","就算","要说","哪怕","福音","集成","几点","小车","有用","超速","客服","不懂","紧张","大人","父母","白天","判断",
        "太棒", "倍感", "媲美", "香薰", "有啥", "心目中", "一回", "旗舰型", "台面", "忘了", "较强", "公众", "总成",
        "路人", "指数", "眼里", "省去", "点头",
        "通透", "一气呵成", "可达", "男人", "度高", "起伏", "一贯", "久坐", "小点", "听听", "成都", "聊天", "衡量",
        "拿捏", "方式", "降低", "多余", "关注",
        "场景", "进行", "驾驶位", "想象", "才能", "泊车", "区别", "理想", "车位",  "更是", "接近",
        "隐藏式", "全程", "蓝牙", "最多", "花里胡哨",
        "突然", "今年", "平顺", "实现", "东北", "无线", "一天", "别人", "比亚迪", "太小", "太高", "贴心", "自带",
        "下面", "对手", "成年人", "自适应", "物品",
        "改善", "块钱", "居然", "态度", "减速", "时代", "环境", "找到", "别人", "汉兰达", "近年", "注意", "内容",
        "突然", "坐车", "极高", "而言", "一切",
        "今天", "地图", "是不是", "较高", "材料", "有时", "状态", "大件", "很小", "高大", "一系列", "暴力", "深得",
        "途中", "牺牲", "小白", "实力", "一条",
        "轻重", "中午", "买单", "困难", "路过", "礼物", "小小的", "足以", "极佳", "靠近", "较低", "就喜欢", "那里",
        "起飞", "太小", "进出", "静音", "透明",
        "欣赏", "不来", "成绩", "很强", "大量", "注意", "良好", "暂时", "自带", "总的来说", "千瓦", "全程", "决定",
        "范围", "下面", "最低", "路程", "始终",
        "适中", "很多人", "屁股", "信心", "类型", "大多数", "最主要", "测试", "车位", "十几万", "环境", "气息", "随便",
        "开口", "评价", "复杂", "夏季", "放下",
        "一年", "很久", "织物", "鸿蒙", "接近", "普通", "一辆", "前后", "压抑", "毛病", "相当于", "总的来说", "在线",
        "之类", "太高", "很久", "逻辑", "略显",
        "看电影", "一拳", "说话", "电话", "时速", "地盘", "大约", "上来", "不输", "比起", "附近", "4s", "韶华", "结合",
        "没用过", "没话说", "较好", "哈哈哈",
        "跑起来", "更大", "怎么说", "第一个", "心情", "快速", "夏季", "我家", "不限", "轻轻松松", "公司", "比起",
        "地图", "区域", "因素", "宽度", "听说",
        "很久", "范围内", "关于", "上坡", "两次", "播放", "至于", "一切", "根据", "很低", "不输", "多个", "指向",
        "前段", "半小时", "调好", "腰部", "节省",
        "较低", "点亮", "留下", "企业", "网易", "联系", "电能", "最开始", "多点", "一圈", "仿佛",  "充分",
        "样子", "暂时", "及时", "突然", "活动",
        "理解", "追求", "关键", "没什么", "任何", "预期", "前后", "变得", "尤为", "想对", "比如",  "哈哈",
        "过程", "没深刻", "想对", "相对", "为了",
        "不同", "开车", "一比一",  "不论是", "令人",
         "也挺", "有时候", "最终", "不好", "没得", "一段", "不大", "太大", "很快", "左右", "其它", "本身",
        "周末", "最后", "很高", "合适", "值得", "没啥", "几个", "更好", "没想到", "一起", "最好", "前面", "一大",
        "两个", "满满",  "最大", "更多", "在路上", "未来", "不一样", "最重要", "要是", "是因为", "都会",
        "给力", "还要",  "终身免费", "一毛钱", "上次", "具体", "真心", "相信", "妥妥", "身边", "人员", "存在",
        "真正", "哪个", "一会", "有限", "可能会", "方位", "不怕", "真实", "不在话下", "成为", "每个人", "都喜欢",
        "无论", "过来",  "准备", "之内", "对了", "一眼", "每月", "尝试", "很难", "不可", "有余", "永远", "随身",
        "一路", "计划",  "一点点", "平常", "两三个",  "不远", "参考", "中意", "因人而异", "证明",
        "都喜欢",  "每周", "自如", "正在", "有一个", "说明", "一半", "有人", "为主", "一位", "那天", "成熟",
        "不说", "两年", "两天", "有效", "慢慢", "仁者见仁", "看看", "有所", "不再", "每一次", "有没有", "怎么样",
        "一块", "上下", "意外", "一趟", "上次", "特别", "每一个", "随意", "一致", "两侧", "多项", "愿意", "自家",
        "得到", "第一眼",
          "不愧", "并不", "感觉到",  "妥妥",  "也挺", "很快", "值得", "需要",
        "一家人", "更好", "几个",  "小型", "前排", "有时候", "最喜欢", "太大", "不大", "最好", "最大",
        "很高",  "前面",  "长时间", "上车", "下车", "体验",
         "适合", "在意", "最后",  "左右",  "丰富", "平均", "挺好",  "时间", "效果",
        "最终",    "满满", "算是",  "购买",
          "省心",  "体重", "特点", "很长", "很慢", "太快", "考虑到", "杠杠的", "在路上", "优点",
        "缺点", "公里", "挑剔", "强烈", "突出", "真心", "人员",  "好多", "放弃", "其中", "很难",
         "太慢", "超出",  "其它", "要是", "清晰", "无疑", "可能会",   "放置",
        "得益",  "有余", "节约",  "标准", "极致", "舍得",  "迅速", "开启", "考究",
        "角度",  "超级",  "极限", "硬朗",  "自然", "可惜",
        "绰绰有余", "接触", "美式", "遗憾", "来回", "得到", "超高", "当下", "自从",  "超大",
         "一旦", "明白", "发挥", "看见",  "毋庸置疑", "只不过",  "牌子",  "不简单",
        "摸着", "我有",  "立马", "厚道", "膝盖", "源源不断", "下载", "指令", "疲劳", "期待",
          "期待", "冬天",   "牛逼",  "眼光", "一股", "膝盖", "果然", "既然",
        "不久", "大多", "刺激", "内心", "回去", "看法",  "鸡柳",  "补充",  "天使之翼", "源源不断",
        "流量",  "偏高", "海豚", "我会",  "伸展", "阳光",   "拉风", "立马",
         "季节", "腰酸背痛", "风景线", "不便", "爆棚", "歌曲", "路途", "早晚", "较快", "一场", "太远",
        "一分钱", "见仁见智",  "吃饭", "欣慰", "玩玩", "道理", "自我", "万一", "那边",  "爽快",
        "要强", "来得", "无所谓",  "缝隙", "很香", "喜好", "换车", "分享", "温柔", "机会",
          "成员",  "对话", "动手", "看车", "搞定", "夸张", "坐满",  "呼吸", "会好",
        "完了", "现象", "既然", "美中不足", "世界", "所在", "蛮好",  "所在", "三十", "踏实", "无所谓", "负担",
        "躺着", "果然", "试试", "两个月", "仅仅",  "呈现", "感谢", "江苏",
        "尴尬", "耐心",  "最强",  "心疼", "千万",  "偶然", "爱上",  "几十", "大定",
        "惬意", "中央","半年","两种","前方","手套","后悔","驾驶者","网上","品牌","优势","强项","8折","靠谱",
        "同学","中途","预警","差别","玩家","青色","公分","膝部","项目","格局","段时间","集中","场合",
        "相当大","奶茶","观点",
        "手指","行业","半年","新版","玩具","身材","统一","前方","旁边","经历","两种","记录","文化","后背","各种各样",
        "既有","有一说一","最初","起到好处","水杯","大众","眼睛",
        "半年","后方","警示","理念","评论","差别","记录","地下","商场","双手","相关","流行","有关",






    ])
    return stopwords

brand = dict({
    '比亚迪': ['宋PLUS新能源', '海豚', '秦PLUS', '宋Pro新能源', '汉', '海豹', '比亚迪e2', '元PLUS', '唐新能源', '秦新能源', '海豹06新能源', '宋MAX新能源', '宋L EV', '海豹06 DM-i'],
    '极氪': ['极氪001', '极氪007', '极氪X'],
    '零跑': ['零跑C11', '零跑C10', '零跑C01'],
    '蔚来': ['蔚来ES6', '蔚来ET5', '蔚来ET5T', '蔚来ES7'],
    '小鹏': ['小鹏P7', '小鹏G3', '小鹏P5', '小鹏G6', '小鹏G9'],
    '大众': ['帕萨特新能源', '途观L新能源', 'ID.4 X', 'ID.6 X', '大众ID.3', 'ID.4 CROZZ', 'ID.6 CROZZ', '探岳GTE插电混动', '迈腾GTE插电混动'],
    '腾势': ['腾势D9'],
    '问界': ['问界M5', '问界M7', '问界M9'],
    '广汽埃安': ['AION Y', 'AION V', 'IQ锐歌', 'AION S'],
    '荣威': ['荣威RX5新能源', '荣威Ei5', '荣威iMAX8新能源', '荣威i6 MAX新能源'],
    '哪吒': ['哪吒U', '哪吒S'],
    '吉利': ['吉利几何C', '吉利几何A', '帝豪L HiP', '帝豪新能源', '吉利几何M6'],
    '岚图': ['岚图FREE', '岚图梦想家'],
    '极狐': ['极狐 阿尔法S(ARCFOX αS)', '极狐 阿尔法T(ARCFOX αT)'],
    '银河': ['银河L7'],
    '护卫舰': ['护卫舰07'],
    '特斯拉': ['Model 3', 'Model Y'],
    'smart': ['smart精灵#1'],
    '豹': ['豹5'],
    '宝马': ['宝马iX3', '宝马i3'],
    '海狮': ['海狮07 EV'],
    '福特': ['福特电马'],
    '领克': ['领克08新能源', '领克09新能源', '领克01新能源', '领克08'],
    '欧拉': ['欧拉好猫', '欧拉闪电猫', '欧拉芭蕾猫'],
    'MG': ['MG4 EV'],
    '理想': ['理想ONE', '理想L7', '理想L9', '理想L8', '理想L6'],
    '长安': ['长安UNI-K 智电iDD', '畅巡', '长安欧尚Z6新能源', '长安UNI-V 智电iDD', '逸动新能源'],
    '飞凡': ['飞凡R7', '飞凡MARVEL R', '飞凡F7'],
    '深蓝': ['深蓝S07'],
    '智己': ['智己LS6', '智己LS7'],
    '别克': ['别克E5'],
    '启辰': ['启辰D60EV'],
    '魏牌': ['魏牌 摩卡新能源', '魏牌 蓝山', '魏牌 拿铁DHT-PHEV'],
    '小米': ['小米SU7'],
    '北京': ['北京EU5'],
    '丰田': ['丰田bZ3', '一汽丰田bZ4X', '广汽丰田bZ4X'],
    '哈弗': ['哈弗枭龙MAX', '哈弗H6新能源', '哈弗二代大狗新能源'],
    '名爵': ['名爵HS新能源'],
    '奥迪': ['奥迪Q4 e-tron', '奥迪Q5 e-tron'],
    '合创': ['合创Z03'],
    '坦克': ['坦克400新能源', '坦克500新能源'],
    '沃尔沃': ['沃尔沃S90插电式混动'],
    '红旗': ['红旗E-QM5'],
    '威马': ['威马E.5'],
    '传祺': ['传祺E8新能源'],
    '本田': ['本田e:NS1', '本田CR-V新能源', '皓影新能源'],
    '东风本田': ['e:NP1 极湃1', '东风本田M-NV'],
    '东风': ['东风风神E70'],
    '奇瑞': ['艾瑞泽e'],
    '奔驰': ['奔驰E级新能源', '奔驰EQB'],
    '奔腾': ['奔腾NAT'],
    '创维': ['创维EV6'],
    '思皓': ['思皓爱跑']
})

# 1. 数据预处理
def preprocess_reviews(reviews_df):
    """
    处理用户评论数据
    :param reviews_df: 包含用户评论的DataFrame
    :return: 处理后的数据
    """
    # reviews_df有列: model, review_text
    processed_data = []
    error_count = 0
    total_reviews = 0

    for idx, row in reviews_df.iterrows():
        try:
            model = row['model']
            review_texts = row['review_text']  # 假设这是一个包含8条评论的列表

            max_text_length = 999
            for text in review_texts:
                total_reviews += 1
                try:
                    text = text[:max_text_length] if len(text) > max_text_length else text

                    # 分词
                    try:
                        request = CommonRequest()
                        request.set_domain('alinlp.cn-hangzhou.aliyuncs.com')
                        request.set_version('2020-06-29')
                        request.set_action_name('GetWsChGeneral')
                        request.add_query_param('ServiceCode', 'alinlp')
                        request.add_query_param('Text', text)
                        request.add_query_param('TokenizerId', 'GENERAL_CHN')
                        request.add_query_param('OutType', '0')

                        response = client.do_action_with_exception(request)
                        resp_obj = json.loads(response)

                        # 解析分词结果，只删除标签列表中只有"基本词-中文"的词
                        words = []
                        result_data = json.loads(resp_obj['Data'])
                        for word_item in result_data['result']:
                            # 检查tags是否为空
                            if not word_item['tags']:
                                pass
                            # 如果tags不为空，检查是否只有一个标签且为"基本词-中文"
                            elif len(word_item['tags']) > 1 or word_item['tags'][0] != "基本词-中文":
                                words.append(word_item['word'])
                            # 如果只有"基本词-中文"标签，则跳过这个词
                    except Exception as e:
                        print(f"分词API错误(行 {idx}, 文本长度 {len(text)}): {e}")
                        words = []  # 分词失败时使用空列表

                    # 情感分析
                    try:
                        sentiment_request = CommonRequest()
                        sentiment_request.set_domain('alinlp.cn-hangzhou.aliyuncs.com')
                        sentiment_request.set_version('2020-06-29')
                        sentiment_request.set_action_name('GetSaChGeneral')
                        sentiment_request.add_query_param('ServiceCode', 'alinlp')
                        sentiment_request.add_query_param('Text', text)

                        sentiment_response = client.do_action_with_exception(sentiment_request)
                        sentiment_obj = json.loads(sentiment_response)
                        sentiment_data = json.loads(sentiment_obj['Data'])
                        sentiment = sentiment_data['result']['sentiment']
                        positive_prob = sentiment_data['result']['positive_prob']
                        negative_prob = sentiment_data['result']['negative_prob']
                        neutral_prob = sentiment_data['result']['neutral_prob']
                    except Exception as e:
                        print(f"情感分析API错误(行 {idx}, 文本长度 {len(text)}): {e}")
                        sentiment = "错误"
                        positive_prob = 0.0
                        negative_prob = 0.0
                        neutral_prob = 0.0

                    processed_data.append({
                        'model': model,
                        'words': words,
                        'sentiment': sentiment,
                        'text': text,
                        'positive_prob': positive_prob,
                        'negative_prob': negative_prob,
                        'neutral_prob': neutral_prob
                    })

                    # 添加进度提示
                    if total_reviews % 10 == 0:
                        print(f"已处理 {total_reviews} 条评论，跳过 {error_count} 条错误")

                except Exception as e:
                    error_count += 1
                    print(f"处理单条评论错误(行 {idx}): {e}")
                    # 继续处理下一条评论
                    continue
        except Exception as e:
            print(f"处理数据行错误(行 {idx}): {e}")
            # 继续处理下一行数据
            continue

    print(f"处理完成: 总共处理 {total_reviews} 条评论，跳过 {error_count} 条错误")

    # 如果没有成功处理任何数据，返回空DataFrame
    if not processed_data:
        print("警告: 没有成功处理任何数据!")
        return pd.DataFrame(columns=['model', 'words', 'sentiment', 'text',
                                     'positive_prob', 'negative_prob', 'neutral_prob'])

    return pd.DataFrame(processed_data)


def identify_part_of_speech(word_list, client, required_ratio={'NN': 0.6, 'JJ': 0.4}):
    """
    使用阿里云API识别词性并按照指定比例筛选属性
    :param word_list: 待检查词性的单词列表
    :param client: 阿里云客户端
    :param required_ratio: 所需的词性比例，如 {'n': 0.7, 'a': 0.3} 表示70%名词，30%形容词
    :return: 词性字典和按比例筛选后的词列表
    """
    print(f"正在识别 {len(word_list)} 个词的词性...")

    # 对词列表进行分批处理，减少API调用次数
    batch_size = 100  # 每次处理100个词
    pos_dict = {}

    # 批量处理词性识别
    for i in range(0, len(word_list), batch_size):
        batch_words = word_list[i:min(i + batch_size, len(word_list))]

        # 将批次词合并成一个文本，用空格分隔
        combined_text = ' '.join(batch_words)

        try:
            request = CommonRequest()
            request.set_domain('alinlp.cn-hangzhou.aliyuncs.com')
            request.set_version('2020-06-29')
            request.set_action_name('GetPosChGeneral')  # 使用词性标注API
            request.add_query_param('ServiceCode', 'alinlp')
            request.add_query_param('Text', combined_text)

            response = client.do_action_with_exception(request)
            resp_obj = json.loads(response)

            if 'Data' in resp_obj:
                result_data = json.loads(resp_obj['Data'])

                # 处理API返回的词性标注结果
                if 'result' in result_data:
                    for item in result_data['result']:
                        word = item['word']
                        pos = item['pos']  # 词性代码: n-名词, a-形容词, v-动词等

                        # 只关注名词和形容词
                        if pos in ['NN', 'JJ']:
                            pos_dict[word] = pos

        except Exception as e:
            print(f"词性识别API错误: {e}")
            continue

    # 打印词性统计
    noun_count = sum(1 for pos in pos_dict.values() if pos == 'NN')
    adj_count = sum(1 for pos in pos_dict.values() if pos == 'JJ')
    print(f"词性识别结果: 名词 {noun_count} 个, 形容词 {adj_count} 个")

    # 将词按词性分组
    nouns = [word for word, pos in pos_dict.items() if pos == 'NN']
    adjs = [word for word, pos in pos_dict.items() if pos == 'JJ']

    # 按照指定比例选择词
    total_needed = len(word_list)
    nouns_needed = int(total_needed * required_ratio['NN'])
    adjs_needed = int(total_needed * required_ratio['JJ'])

    # 调整比例以确保总数一致
    if nouns_needed + adjs_needed < total_needed:
        nouns_needed += (total_needed - (nouns_needed + adjs_needed))

    # 如果某一类词不足，从另一类补充
    if len(nouns) < nouns_needed:
        nouns_actual = len(nouns)
        adjs_needed += (nouns_needed - nouns_actual)
        nouns_needed = nouns_actual

    if len(adjs) < adjs_needed:
        adjs_actual = len(adjs)
        nouns_needed += (adjs_needed - adjs_actual)
        adjs_needed = adjs_actual

    # 选择所需数量的词
    selected_nouns = nouns[:nouns_needed] if nouns_needed <= len(nouns) else nouns
    selected_adjs = adjs[:adjs_needed] if adjs_needed <= len(adjs) else adjs

    selected_words = selected_nouns + selected_adjs

    # 计算实际比例
    actual_ratio = {
        'NN': len(selected_nouns) / max(1, len(selected_words)),
        'JJ': len(selected_adjs) / max(1, len(selected_words))
    }

    print(
        f"词性筛选后: 名词 {len(selected_nouns)}个 ({actual_ratio['NN']:.2%}), 形容词 {len(selected_adjs)}个 ({actual_ratio['JJ']:.2%})")

    return pos_dict, selected_words

# 2.加载预处理数据
# Modified function to load processed data by brand
def load_processed_data_by_brand(file_path='processed_reviews.xlsx', brand_mapping=brand):
    """
    加载预处理后的数据并按品牌分组
    :param file_path: Excel文件路径
    :param brand_mapping: 品牌到车型的映射字典
    :return: 按品牌分组的DataFrame
    """
    print(f"加载预处理数据: {file_path}")
    processed_df = pd.read_excel(file_path)

    # 获取停用词表
    stopwords = get_stopwords()
    print(f"加载了 {len(stopwords)} 个停用词")

    # 创建反向映射：车型到品牌
    model_to_brand = {}
    for brand, models in brand_mapping.items():
        for model in models:
            model_to_brand[model] = brand

    # 添加品牌列
    processed_df['brand'] = processed_df['model'].map(model_to_brand)

    # 统计每个品牌的数据量
    brand_counts = processed_df['brand'].value_counts()
    print("\n===== 各品牌数据量统计 =====")
    for brand, count in brand_counts.items():
        print(f"品牌: {brand} - 数据量: {count}条")
    print("==========================\n")

    # 将字符串形式的词列表转换回Python列表
    if 'words' in processed_df.columns:
        # 检查第一行的类型来决定如何处理
        if processed_df.shape[0] > 0:  # 确保有数据
            first_words = processed_df['words'].iloc[0]

            # 如果已经是列表类型，不需要转换
            if not isinstance(first_words, list):
                # 如果是字符串，尝试转换成列表
                def parse_words(words_str):
                    if pd.isna(words_str):
                        return []

                    if isinstance(words_str, str):
                        # 尝试使用eval安全转换，如果格式像Python列表
                        if words_str.startswith('[') and words_str.endswith(']'):
                            try:
                                return eval(words_str)
                            except:
                                # 如果eval失败，使用字符串分割
                                clean_str = words_str.strip('[]').replace("'", "").replace('"', '')
                                return [w.strip() for w in clean_str.split(',')]
                        else:
                            # 如果不是列表格式，作为单个单词返回
                            return [words_str]
                    elif isinstance(words_str, list):
                        return words_str
                    else:
                        return []

                processed_df['words'] = processed_df['words'].apply(parse_words)
                print("已将字符串形式的words列表转换为Python列表")

        # 确认转换是否成功
        if processed_df.shape[0] > 0:
            print(f"第一行words示例: {processed_df['words'].iloc[0][:5]}...")

        # 对每行数据中的words列表进行停用词过滤
        def filter_stopwords(word_list):
            if not isinstance(word_list, list):
                return []
            return [word for word in word_list if word not in stopwords and word.strip() != '']

        # 应用停用词过滤
        processed_df['words'] = processed_df['words'].apply(filter_stopwords)

        # 计算过滤前后的单词总数以验证效果
        words_before = processed_df['words'].explode().shape[0]
        stopwords_filtered = sum(1 for words in processed_df['words'] for word in words if word in stopwords)
        print(f"停用词过滤完成，已过滤 {stopwords_filtered} 个停用词")
    else:
        print("警告: 数据中没有'words'列!")

    return processed_df


# 3. 提取车型属性
# 修改后的函数：提取品牌属性而不是车型属性
def extract_brand_attributes(processed_df, min_frequency=2, min_word_length=2):
    """
    提取品牌关联属性，对负面属性采用更低的过滤阈值，并按照70%名词、30%形容词的比例筛选
    :param processed_df: 处理后的数据
    :param min_frequency: 最小出现频率
    :param min_word_length: 最小词长度
    :return: 品牌属性字典
    """
    print("正在提取品牌属性（按词性比例筛选）...")
    brand_attributes = {}
    stopwords = get_stopwords()

    for brand in processed_df['brand'].unique():
        if brand not in ['比亚迪', '特斯拉', '吉利', '理想', '广汽埃安', '问界', '零跑', '蔚来', '小鹏', '小米']:
            continue
        brand_df = processed_df[processed_df['brand'] == brand]

        # 收集所有词及其情感
        word_sentiments = {}
        for _, row in brand_df.iterrows():
            positive_prob = row['positive_prob']
            # 将情感标签"正面"和"负面"转换为"positive"和"negative"
            sentiment_key = 'positive' if positive_prob >= 0.88 else 'negative'
            # 权重根据概率值计算
            weight = row['positive_prob'] if sentiment_key == 'positive' else row[
                'negative_prob'] if sentiment_key == 'negative' else row['neutral_prob']

            for word in row['words']:
                # 过滤掉停用词和短词
                if word in stopwords:
                    continue
                if word not in stopwords and len(word) >= min_word_length:
                    if word not in word_sentiments:
                        word_sentiments[word] = {'positive': 0, 'negative': 0, 'neutral': 0, 'weighted_positive': 0,
                                                 'weighted_negative': 0, 'weighted_neutral': 0}
                    word_sentiments[word][sentiment_key] += 1
                    word_sentiments[word][f'weighted_{sentiment_key}'] += weight

        # 过滤低频词，但对负面属性使用更低的阈值
        filtered_attributes = {}
        for word, counts in word_sentiments.items():
            total = counts['positive'] + counts['negative'] + counts['neutral']

            # 判断主要情感
            if counts['positive'] >= counts['negative'] and counts['positive'] >= counts['neutral']:
                main_sentiment = 'positive'
                strength = counts['positive'] / total
                # 正面属性使用正常阈值
                min_freq_threshold = min_frequency
            elif counts['negative'] >= counts['positive'] and counts['negative'] >= counts['neutral']:
                main_sentiment = 'negative'
                strength = counts['negative'] / total
                # 负面属性使用更低的阈值，确保更多负面属性被保留
                min_freq_threshold = max(1, min_frequency - 1)
            else:
                main_sentiment = 'neutral'
                strength = counts['neutral'] / total
                min_freq_threshold = min_frequency

            # 只保留达到阈值的词
            if total >= min_freq_threshold:
                # 计算情感得分 (-1到1范围)
                sentiment_score = (counts['positive'] - counts['negative']) / total

                filtered_attributes[word] = {
                    'count': total,
                    'sentiment': main_sentiment,
                    'strength': strength,
                    'sentiment_score': sentiment_score,
                    'positive_count': counts['positive'],
                    'negative_count': counts['negative'],
                    'neutral_count': counts['neutral'],
                    'weighted_positive': counts['weighted_positive'],
                    'weighted_negative': counts['weighted_negative'],
                    'weighted_neutral': counts['weighted_neutral']
                }

        # 对正向属性进行额外检查，如果positive_prob小于0.83，将其重新分类为负向
        for word, attr_info in list(filtered_attributes.items()):
            if attr_info['sentiment'] == 'positive':
                # 计算该词的正向概率加权平均值
                total_weight = attr_info['weighted_positive'] + attr_info['weighted_negative'] + attr_info[
                    'weighted_neutral']
                if total_weight > 0:
                    avg_positive_prob = attr_info['weighted_positive'] / total_weight

                    # 如果正向概率小于0.83，将其改为负向
                    if avg_positive_prob < 0.83:
                        attr_info['sentiment'] = 'negative'
                        print(f"    - 属性 '{word}' 被重新分类为负向 (正向概率: {avg_positive_prob:.2f})")

        # 进行词性分析并按比例筛选
        attribute_words = list(filtered_attributes.keys())
        if attribute_words:
            # 进行词性识别和筛选
            pos_dict, selected_words = identify_part_of_speech(
                attribute_words,
                client,
                required_ratio={'NN': 0.6, 'JJ': 0.4}  # 设置70%名词，30%形容词的比例
            )

            # 更新filtered_attributes，只保留筛选后的词
            filtered_attributes_pos = {word: filtered_attributes[word] for word in selected_words if
                                       word in filtered_attributes}

            # 为每个属性添加词性信息
            for word in filtered_attributes_pos:
                filtered_attributes_pos[word]['pos'] = pos_dict.get(word, 'unknown')

            brand_attributes[brand] = filtered_attributes_pos
        else:
            brand_attributes[brand] = {}

        print(f"  品牌 '{brand}' 提取了 {len(brand_attributes[brand])} 个属性")

        # 统计正负面属性数量
        positive_count = sum(1 for attr, info in brand_attributes[brand].items() if info['sentiment'] == 'positive')
        negative_count = sum(1 for attr, info in brand_attributes[brand].items() if info['sentiment'] == 'negative')
        print(f"    - 正面属性: {positive_count}个, 负面属性: {negative_count}个")

        # 统计词性分布
        noun_count = sum(1 for attr, info in brand_attributes[brand].items() if info.get('pos') == 'NN')
        adj_count = sum(1 for attr, info in brand_attributes[brand].items() if info.get('pos') == 'JJ')
        print(f"    - 名词: {noun_count}个 ({noun_count / max(1, len(brand_attributes[brand])):.2%}), "
              f"形容词: {adj_count}个 ({adj_count / max(1, len(brand_attributes[brand])):.2%})")

    return brand_attributes


# 4. 计算属性中心度
# 修改后的函数：计算品牌属性中心度
def calculate_brand_centrality(brand_attributes, processed_df):
    """
    计算品牌属性的中心度
    :param brand_attributes: 品牌属性字典
    :param processed_df: 所有评论数据
    :return: 带中心度的品牌属性字典
    """
    print("正在计算品牌属性中心度...")

    # 添加进度条显示品牌处理进度
    for brand in tqdm(brand_attributes.keys(), desc="计算品牌属性中心度"):
        brand_df = processed_df[processed_df['brand'] == brand]
        total_reviews = len(brand_df)

        # 创建词在文档中的频率计数
        doc_counts = {}
        for _, row in brand_df.iterrows():
            words_in_doc = set(row['words'])  # 确保每个词在一个文档中只计算一次
            for word in words_in_doc:
                doc_counts[word] = doc_counts.get(word, 0) + 1

        # 添加属性处理进度条
        for attr, info in tqdm(brand_attributes[brand].items(), desc=f"处理 {brand} 属性", leave=False):
            # 计算归一化加权度中心度 (NWCD)
            # 考虑词的出现频率和文档频率
            if attr in doc_counts:
                doc_freq = doc_counts[attr]
                tf = info['count']  # 词频
                idf = np.log(total_reviews / (1 + doc_freq))  # 逆文档频率(添加1以避免除以0)
                tf_idf = tf * idf  # TF-IDF得分

                # 归一化中心度
                nwcd = tf_idf / total_reviews
                brand_attributes[brand][attr]['nwcd'] = nwcd
                brand_attributes[brand][attr]['doc_freq'] = doc_freq
                brand_attributes[brand][attr]['tf_idf'] = tf_idf
            else:
                # 默认值
                brand_attributes[brand][attr]['nwcd'] = 0
                brand_attributes[brand][attr]['doc_freq'] = 0
                brand_attributes[brand][attr]['tf_idf'] = 0

    # 检测和移除离群值
    for brand in tqdm(brand_attributes.keys(), desc="处理离群值"):
        if len(brand_attributes[brand]) > 0:
            # 计算NWCD的四分位数
            nwcd_values = [info['nwcd'] for info in brand_attributes[brand].values()]
            q1 = np.percentile(nwcd_values, 25)
            q3 = np.percentile(nwcd_values, 75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr

            # 找到离群值
            outliers = [attr for attr, info in brand_attributes[brand].items() if info['nwcd'] > upper_bound]
            print(f"  品牌 '{brand}' 检测到 {len(outliers)} 个中心度离群值")

            # 移除离群值
            for outlier in outliers:
                del brand_attributes[brand][outlier]

    return brand_attributes


# 5. 确定属性间关联
# 修改后的函数：识别品牌属性间关联
def identify_brand_attribute_connections(processed_df, brand_attributes):
    """
    识别品牌属性之间的关联
    :param processed_df: 处理后的数据
    :param brand_attributes: 品牌属性字典
    :return: 关联图
    """
    print("正在识别品牌属性间关联...")
    connections = {}

    # 添加进度条显示品牌处理进度
    for brand in tqdm(brand_attributes.keys(), desc="识别品牌属性关联"):
        connections[brand] = {}
        brand_df = processed_df[processed_df['brand'] == brand]

        # 创建属性列表
        attributes = list(brand_attributes[brand].keys())

        # 初始化连接计数器
        for attr1 in attributes:
            connections[brand][attr1] = {}
            for attr2 in attributes:
                if attr1 != attr2:
                    connections[brand][attr1][attr2] = 0

        # 计算共现次数（添加进度条）
        for _, row in tqdm(brand_df.iterrows(), desc=f"处理 {brand} 共现", leave=False):
            words_in_review = set(word for word in row['words'] if word in attributes)
            for attr1 in words_in_review:
                for attr2 in words_in_review:
                    if attr1 != attr2:
                        connections[brand][attr1][attr2] += 1

        print(f"  品牌 '{brand}' 已计算属性关联")

    return connections


# 6. 修改网络构建函数，增强负面属性的表现
# 修改后的函数：构建品牌网络图
# Modified build_brand_network function to ensure compatibility with the visualization changes
def build_brand_network(brand, brand_attributes, connections, min_edge_weight=0.8, top_attributes=100):
    """
    构建优化的品牌关联网络，增强节点数据以支持改进的可视化，并维持70%名词、30%形容词的比例
    :param brand: 品牌名
    :param brand_attributes: 品牌属性字典
    :param connections: 属性间连接
    :param min_edge_weight: 最小边权重
    :param top_attributes: 显示前N个最重要的属性
    :return: NetworkX图对象
    """
    print(f"正在构建 '{brand}' 的网络图 (词性比例优化版)...")
    G = nx.Graph()

    # 获取所有属性
    all_attrs = brand_attributes[brand]

    # 分离正面和负面属性
    positive_attrs = {attr: info for attr, info in all_attrs.items() if info['sentiment'] == 'positive'}
    negative_attrs = {attr: info for attr, info in all_attrs.items() if info['sentiment'] == 'negative'}
    neutral_attrs = {attr: info for attr, info in all_attrs.items() if info['sentiment'] == 'neutral'}

    # 分离名词和形容词属性
    noun_attrs = {attr: info for attr, info in all_attrs.items() if info.get('pos') == 'NN'}
    adj_attrs = {attr: info for attr, info in all_attrs.items() if info.get('pos') == 'JJ'}

    print(
        f"  总属性数: {len(all_attrs)}, 正面: {len(positive_attrs)}, 负面: {len(negative_attrs)}, 中性: {len(neutral_attrs)}")
    print(f"  词性分布: 名词: {len(noun_attrs)}个, 形容词: {len(adj_attrs)}个")

    # 确保有一个合理的正负面属性比例 - 更平衡的展示
    target_negative_ratio = 0.3  # 增加目标负面属性比例从0.3到0.4
    min_negative_count = min(len(negative_attrs), max(5, int(top_attributes * target_negative_ratio)))
    target_neutral_ratio = 0.1  # 目标中性属性比例
    min_neutral_count = min(len(neutral_attrs), max(3, int(top_attributes * target_neutral_ratio)))

    # 剩余槽位分配给正面属性
    remaining_slots = top_attributes - min_negative_count - min_neutral_count

    # 确保词性比例符合要求: 70%名词, 30%形容词
    target_noun_ratio = 0.6
    target_adj_ratio = 0.4

    # 计算正负中各情感类别所需的名词和形容词数量
    neg_nouns_needed = int(min_negative_count * target_noun_ratio)
    neg_adjs_needed = min_negative_count - neg_nouns_needed

    neu_nouns_needed = int(min_neutral_count * target_noun_ratio)
    neu_adjs_needed = min_neutral_count - neu_nouns_needed

    pos_nouns_needed = int(remaining_slots * target_noun_ratio)
    pos_adjs_needed = remaining_slots - pos_nouns_needed

    # 处理负面属性 - 按中心度排序，同时考虑词性比例
    neg_nouns = {attr: info for attr, info in negative_attrs.items() if info.get('pos') == 'NN'}
    neg_adjs = {attr: info for attr, info in negative_attrs.items() if info.get('pos') == 'JJ'}

    sorted_neg_nouns = sorted(neg_nouns.items(), key=lambda x: x[1]['nwcd'], reverse=True)
    sorted_neg_adjs = sorted(neg_adjs.items(), key=lambda x: x[1]['nwcd'], reverse=True)

    # 调整需求数量以适应实际情况
    neg_nouns_needed = min(len(sorted_neg_nouns), neg_nouns_needed)
    neg_adjs_needed = min(len(sorted_neg_adjs), neg_adjs_needed)

    if neg_nouns_needed < int(min_negative_count * target_noun_ratio):
        # 名词不足，尝试补充更多形容词
        additional_adjs = int(min_negative_count * target_noun_ratio) - neg_nouns_needed
        neg_adjs_needed = min(len(sorted_neg_adjs), neg_adjs_needed + additional_adjs)

    selected_neg_nouns = dict(sorted_neg_nouns[:neg_nouns_needed])
    selected_neg_adjs = dict(sorted_neg_adjs[:neg_adjs_needed])
    selected_negative = {**selected_neg_nouns, **selected_neg_adjs}

    # 同样处理中性属性
    neu_nouns = {attr: info for attr, info in neutral_attrs.items() if info.get('pos') == 'NN'}
    neu_adjs = {attr: info for attr, info in neutral_attrs.items() if info.get('pos') == 'JJ'}

    sorted_neu_nouns = sorted(neu_nouns.items(), key=lambda x: x[1]['nwcd'], reverse=True)
    sorted_neu_adjs = sorted(neu_adjs.items(), key=lambda x: x[1]['nwcd'], reverse=True)

    neu_nouns_needed = min(len(sorted_neu_nouns), neu_nouns_needed)
    neu_adjs_needed = min(len(sorted_neu_adjs), neu_adjs_needed)

    if neu_nouns_needed < int(min_neutral_count * target_noun_ratio):
        additional_adjs = int(min_neutral_count * target_noun_ratio) - neu_nouns_needed
        neu_adjs_needed = min(len(sorted_neu_adjs), neu_adjs_needed + additional_adjs)

    selected_neu_nouns = dict(sorted_neu_nouns[:neu_nouns_needed])
    selected_neu_adjs = dict(sorted_neu_adjs[:neu_adjs_needed])
    selected_neutral = {**selected_neu_nouns, **selected_neu_adjs}

    # 处理正面属性
    pos_nouns = {attr: info for attr, info in positive_attrs.items() if info.get('pos') == 'NN'}
    pos_adjs = {attr: info for attr, info in positive_attrs.items() if info.get('pos') == 'JJ'}

    sorted_pos_nouns = sorted(pos_nouns.items(), key=lambda x: x[1]['nwcd'], reverse=True)
    sorted_pos_adjs = sorted(pos_adjs.items(), key=lambda x: x[1]['nwcd'], reverse=True)

    pos_nouns_needed = min(len(sorted_pos_nouns), pos_nouns_needed)
    pos_adjs_needed = min(len(sorted_pos_adjs), pos_adjs_needed)

    if pos_nouns_needed < int(remaining_slots * target_noun_ratio):
        additional_adjs = int(remaining_slots * target_noun_ratio) - pos_nouns_needed
        pos_adjs_needed = min(len(sorted_pos_adjs), pos_adjs_needed + additional_adjs)

    selected_pos_nouns = dict(sorted_pos_nouns[:pos_nouns_needed])
    selected_pos_adjs = dict(sorted_pos_adjs[:pos_adjs_needed])
    selected_positive = {**selected_pos_nouns, **selected_pos_adjs}

    # 合并选择的属性
    selected_attrs = {**selected_negative, **selected_neutral, **selected_positive}

    # 计算词性比例
    total_nouns = sum(1 for attr, info in selected_attrs.items() if info.get('pos') == 'NN')
    total_attrs = len(selected_attrs)
    noun_ratio = total_nouns / total_attrs if total_attrs > 0 else 0
    adj_ratio = 1 - noun_ratio

    print(f"  选择了 {len(selected_attrs)} 个属性，其中名词: {total_nouns}个 ({noun_ratio:.2%}), "
          f"形容词: {total_attrs - total_nouns}个 ({adj_ratio:.2%})")

    # 计算每个属性的连接数
    attr_connections = {}
    for attr1 in selected_attrs:
        attr_connections[attr1] = 0
        for attr2 in selected_attrs:
            if attr1 != attr2 and attr1 in connections[brand] and attr2 in connections[brand][attr1]:
                if connections[brand][attr1][attr2] > 0:
                    attr_connections[attr1] += 1

    # 找出最大和最小连接数，用于归一化
    max_connections = max(attr_connections.values()) if attr_connections else 1
    min_connections = min(attr_connections.values()) if attr_connections else 0

    # 添加属性节点 - 使用更精细的大小缩放
    for attr, info in selected_attrs.items():
        # 属性中心度
        nwcd = info.get('nwcd', 0.01)  # 避免0值

        # 获取该属性的连接数
        connection_count = attr_connections.get(attr, 0)

        # 归一化连接数 (0到1范围)
        if max_connections > min_connections:
            norm_connections = (connection_count - min_connections) / (max_connections - min_connections)
        else:
            norm_connections = 0.5

        # 使用连接数和中心度的组合来确定节点大小
        # 节点大小取决于属性出现频率、中心度和连接数的加权组合
        count_weight = 0.4  # 词频权重
        nwcd_weight = 0.2  # 中心度权重
        connection_weight = 0.4  # 连接数权重 - 新增

        # 对词频和中心度进行归一化
        normalized_count = np.log1p(info['count'])  # 对数变换减少极端值影响
        normalized_nwcd = nwcd * 100  # 缩放中心度

        # 计算最终大小 - 增加连接数的影响
        size = (count_weight * normalized_count +
                nwcd_weight * normalized_nwcd +
                connection_weight * (norm_connections * 10 + 1)) * 4.0  # 增加系数使节点整体变大

        # 确保最小可见大小
        size = max(16, size)

        # 确保保存情感得分用于渐变色彩 - 强化情感分数差异
        # 根据具体情感类型调整情感分数，使正面和负面更加明显
        raw_sentiment_score = info['sentiment_score']

        # 增强情感得分对比度
        if info['sentiment'] == 'positive':
            # 正面情感得分增强，使其更蓝
            sentiment_score = min(1.0, raw_sentiment_score * 1.3)  # 增强对比度
        elif info['sentiment'] == 'negative':
            # 负面情感得分增强，使其更红
            sentiment_score = max(-1.0, raw_sentiment_score * 1.3)  # 增强对比度
        else:
            # 中性情感保持不变
            sentiment_score = raw_sentiment_score

        G.add_node(attr,
                   type='attribute',
                   size=size,
                   sentiment=info['sentiment'],
                   sentiment_score=sentiment_score,
                   count=info['count'],
                   nwcd=nwcd,
                   pos=info.get('pos', 'unknown'),  # 添加词性信息
                   connection_count=connection_count)  # 保存连接数

    # 智能选择边 - 确保图的可读性
    # 收集所有可能的边及其权重
    potential_edges = []
    selected_attr_keys = list(selected_attrs.keys())

    # 最小连接数和最大连接数
    min_connections = 2  # 每个节点至少需要的连接数
    max_edges = min(300, len(selected_attrs) * 8)  # 增加最大边数，提高连接性

    for i, attr1 in enumerate(selected_attr_keys):
        for attr2 in selected_attr_keys[i + 1:]:
            if attr1 in connections[brand] and attr2 in connections[brand][attr1]:
                weight = connections[brand][attr1][attr2]
                if weight >= min_edge_weight:
                    # 计算边的重要性分数 - 结合权重和节点中心度
                    node1_nwcd = selected_attrs[attr1].get('nwcd', 0)
                    node2_nwcd = selected_attrs[attr2].get('nwcd', 0)
                    # 增强重要节点之间的连接
                    edge_importance = weight * (node1_nwcd + node2_nwcd)

                    potential_edges.append((attr1, attr2, weight, edge_importance))

    # 按重要性得分排序
    potential_edges.sort(key=lambda x: x[3], reverse=True)

    # 确保每个节点至少有min_connections个连接
    node_connections = {node: 0 for node in selected_attr_keys}
    necessary_edges = []

    # 首先添加必要的边确保连接性
    for attr1, attr2, weight, _ in potential_edges:
        if node_connections[attr1] < min_connections or node_connections[attr2] < min_connections:
            necessary_edges.append((attr1, attr2, weight))
            node_connections[attr1] += 1
            node_connections[attr2] += 1

            # 如果所有节点都满足最小连接要求，可以提前停止
            if all(count >= min_connections for count in node_connections.values()):
                break

    # 添加剩余高重要性边，但控制总数
    remaining_edges = []
    for attr1, attr2, weight, _ in potential_edges:
        if (attr1, attr2, weight) not in necessary_edges:
            if len(necessary_edges) + len(remaining_edges) < max_edges:
                remaining_edges.append((attr1, attr2, weight))

    # 合并并添加所有边
    all_edges = necessary_edges + remaining_edges
    for attr1, attr2, weight in all_edges[:max_edges]:
        # 使用原始权重，以便可视化时可以区分边的粗细
        # 增加一个权重系数使边权重值有更明显差异
        adjusted_weight = weight * 1.8  # 增加权重以产生更明显的差异
        G.add_edge(attr1, attr2, weight=adjusted_weight)

    negative_count = sum(1 for node in G.nodes() if G.nodes[node].get('sentiment') == 'negative')
    print(
        f"  添加了 {len(G.nodes())} 个属性节点 (其中负面属性 {negative_count} 个) 和 {len(all_edges[:max_edges])} 条边")

    return G


# 7. 可视化网络图
# 修改visualize_brand_network函数，让连接更多的属性边更粗
# 修复可视化函数中的情感颜色映射问题
# Modified visualize_brand_network function with requested changes
def visualize_brand_network(G, title, save_path=None):
    """
    Enhanced network graph visualization with elegant styling and requested changes:
    1. 增强情感颜色的渐变程度
    2. 根据连接数调整节点大小
    3. 连接多的属性用粗线

    :param G: NetworkX graph object
    :param title: Graph title
    :param save_path: Path to save the image
    """
    print(f"正在可视化 '{title}' 网络图 (优化版)...")

    # 获取可用的中文字体
    chinese_font = setup_fonts()

    # 设置绘图样式 - 更现代的外观
    plt.style.use('seaborn-v0_8-whitegrid')

    # 创建图形 - 使用较大的画布增加分辨率
    fig, ax = plt.subplots(figsize=(18, 15), dpi=100)

    # 设置背景颜色 - 使用浅蓝色作为背景色
    bg_color = '#e9eff5'  # 浅蓝灰色
    fig.patch.set_facecolor(bg_color)  # 图形背景
    ax.set_facecolor('#ffffff')  # 保持图表区域为白色

    # 移除孤立节点，确保所有节点至少有一个连接
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        print(f"  移除 {len(isolated_nodes)} 个孤立节点")
        G.remove_nodes_from(isolated_nodes)

    # 使用更优化的布局 - 调整参数以获得更好的节点分布
    pos = nx.spring_layout(G, k=0.2, iterations=200, seed=42)  # 更精细的布局

    # 准备节点样式
    node_sizes = []
    node_colors = []
    node_borders = []
    border_widths = []

    # 创建更鲜艳的颜色方案 - 深红到白到深蓝的渐变
    # 更强烈的颜色对比
    colors = [(0.9, 0.1, 0.1), (1, 1, 1), (0.1, 0.3, 0.9)]  # 鲜艳的红白蓝渐变
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    # 计算节点的连接数
    node_connection_counts = {}
    for node in G.nodes():
        node_connection_counts[node] = len(list(G.neighbors(node)))

    # 确定连接数的最大值和最小值，用于归一化
    max_connections = max(node_connection_counts.values()) if node_connection_counts else 1
    min_connections = min(node_connection_counts.values()) if node_connection_counts else 1

    # 打印节点的情感属性，用于调试
    print("节点情感属性分布:")
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

    for node in G.nodes():
        sentiment = G.nodes[node].get('sentiment', 'unknown')
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

        # 获取连接数
        connection_count = node_connection_counts[node]

        # 计算节点大小 - 混合使用节点原始大小和连接数作为决定因素
        raw_size = G.nodes[node].get('size', 15)

        # 使用连接数作为调整因子，连接数越多，节点越大
        if max_connections > min_connections:
            connection_factor = 1 + 1.5 * ((connection_count - min_connections) / (max_connections - min_connections))
        else:
            connection_factor = 1.5

        # 计算最终节点大小 - 使用原始大小和连接数调整因子
        if sentiment == 'negative':
            size = 70 * raw_size * connection_factor
        else:
            size = 40 * raw_size * connection_factor
        node_sizes.append(size)

        # 获取情感得分来设置颜色 - 增强情感分数差异
        sentiment_score = G.nodes[node].get('sentiment_score', 0)
        sentiment = G.nodes[node].get('sentiment', 'neutral')

        # 增强颜色差异 - 更鲜艳的颜色
        if sentiment == 'positive':  # 正向 - 粉色调
            # 正向情感得分越高越红，增强红色饱和度
            blue_intensity = min(1.0, 0.4 + 0.6 * sentiment_score)  # 提高基础值和系数
            rgba_color = (0.0, 0.2 * (1 - sentiment_score), blue_intensity, 0.95)  # 降低绿色分量
        elif sentiment == 'negative':  # 负向 - 深红
            # 负向情感得分越高越红，增强红色饱和度
            red_intensity = min(1.0, 0.4 + 0.6 * abs(sentiment_score))  # 提高基础值和系数
            rgba_color = (red_intensity, 0.0, 0.1 * (1 - abs(sentiment_score)), 0.95)  # 降低蓝色分量
        else:  # 中性
            rgba_color = (0.7, 0.7, 0.7, 0.85)  # 灰色

        node_colors.append(rgba_color)

        # 边框颜色 - 基于情感分数但略深
        if sentiment_score < -0.4:  # 负面
            border_color = '#8a2a2a'  # 深红色
        elif sentiment_score > 0.4:  # 正面
            border_color = '#2a4c8a'  # 深蓝色
        else:  # 中性
            border_color = '#5c656d'  # 灰色

        node_borders.append(border_color)

        # 文本颜色
        if sentiment == 'positive' or sentiment == 'negative':
            text_color = 'white'  # 对正负面情感使用白色文本
        else:
            text_color = 'black'  # 中性情况用黑色

        # 存储文本颜色到节点属性
        G.nodes[node]['text_color'] = text_color

        # 边框宽度 - 使用更细的边框
        border_widths.append(1.5)  # 稍微增加边框宽度

    # 准备边的样式 - 使用黑色边和基于连接数的粗细
    edge_widths = []
    edge_alphas = []  # 边的透明度

    for u, v in G.edges():
        # 计算边的粗细 - 基于两个节点的连接数
        source_connections = node_connection_counts[u]
        target_connections = node_connection_counts[v]
        avg_connections = (source_connections + target_connections) / 2

        # 归一化连接数到更广的范围[0.5, 5.0]增强差异
        if max_connections == min_connections:  # 避免除以零
            edge_width = 2.0
        else:
            normalized_connections = (avg_connections - min_connections) / (max_connections - min_connections)
            # 使用非线性映射，让高连接度的边更粗
            edge_width = 0.7 + (normalized_connections ** 0.6) * 4.8  # 提高最大宽度和调整指数

        edge_widths.append(edge_width)

        # 边的透明度 - 连接数越多，透明度越低（越显眼）
        edge_alpha = 0.5 + normalized_connections * 0.45  # 范围[0.5, 0.95]
        edge_alphas.append(edge_alpha)

    # 绘制边 - 使用黑色边，粗细基于连接数
    edges = nx.draw_networkx_edges(G, pos,
                                   width=edge_widths,
                                   alpha=edge_alphas,
                                   edge_color='#000000',  # 黑色边
                                   connectionstyle='arc3,rad=0.05',  # 稍微弯曲的边，避免重叠
                                   ax=ax)

    # 绘制节点 - 自定义颜色和大小
    nodes = nx.draw_networkx_nodes(G, pos,
                                   node_size=node_sizes,
                                   node_color=node_colors,
                                   edgecolors=node_borders,
                                   linewidths=border_widths,
                                   alpha=0.92,  # 稍增透明度
                                   ax=ax)

    # 为属性节点添加标签，直接放在节点上
    for node in G.nodes():
        # 获取节点位置
        x, y = pos[node]
        # 获取节点属性
        node_size = G.nodes[node].get('size', 15)
        connection_count = G.nodes[node].get('connection_count', 0)

        # 使用节点大小和连接数综合计算字体大小
        base_size = 9 + 5 * np.log1p(node_size) / np.log1p(50)  # 基于节点大小的基础字体大小
        conn_factor = 1 + 0.3 * (connection_count / max(max_connections, 1))  # 连接数调整因子
        font_size = base_size * conn_factor  # 最终字体大小

        # 限制字体大小范围
        font_size = min(16, max(9, font_size))

        # 获取之前设置好的文本颜色
        text_color = G.nodes[node].get('text_color', 'black')

        # 直接将文本放在节点上，不使用背景
        plt.text(x, y, node,
                 fontsize=font_size,
                 fontfamily=chinese_font,
                 ha='center', va='center',
                 color=text_color,
                 weight='bold' if G.nodes[node].get('sentiment') in ['positive', 'negative'] else 'normal')

    # 添加颜色条 - 增强颜色范围效果
    norm = plt.Normalize(-1, 1)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical',
                        pad=0.01, fraction=0.02,
                        location='right')
    cbar.set_label('情感分数', fontfamily=chinese_font, fontsize=12, color='#343a40')
    cbar.ax.tick_params(colors='#343a40', labelsize=10)

    # 设置标题 - 使用更优雅的样式
    plt.title(title, fontsize=22, fontfamily=chinese_font, pad=20,
              color='#343a40', fontweight='normal')

    # 设置精确的边界，减少空白区域
    plt.tight_layout(pad=3.0)
    ax.set_axis_off()

    # 添加简单的边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_color('#e9ecef')
    ax.spines['right'].set_color('#e9ecef')
    ax.spines['bottom'].set_color('#e9ecef')
    ax.spines['left'].set_color('#e9ecef')

    # 保存高质量图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"  图像已保存到: {save_path}")

    plt.show()


def export_brand_attributes_to_excel(brand_attributes, output_file='brand_attributes_analysis.xlsx',
                                     max_attrs_per_brand=50):
    """
    将所有品牌的属性信息导出到Excel文件，每个品牌限制最多max_attrs_per_brand个属性，同时保持70%名词和30%形容词的比例
    :param brand_attributes: 品牌属性字典
    :param output_file: 输出Excel文件路径
    :param max_attrs_per_brand: 每个品牌最多展示的属性数量
    :return: 导出的DataFrame
    """
    print(
        f"正在导出所有品牌属性信息到: {output_file} (每个品牌最多{max_attrs_per_brand}个属性，保持70%名词30%形容词比例)")

    # 创建一个空的DataFrame
    df = pd.DataFrame()

    # 目标词性比例
    target_noun_ratio = 0.6
    target_adj_ratio = 0.4

    # 按品牌逐个处理
    for brand_name in sorted(brand_attributes.keys()):
        attributes = brand_attributes[brand_name]

        # 如果是指定的重点品牌，则继续，否则跳过
        if brand_name not in ['比亚迪', '特斯拉', '吉利', '理想', '广汽埃安', '问界', '零跑', '蔚来', '小鹏', '小米']:
            continue

        # 分离名词和形容词属性
        noun_attrs = {attr: info for attr, info in attributes.items() if info.get('pos') == 'NN'}
        adj_attrs = {attr: info for attr, info in attributes.items() if info.get('pos') == 'JJ'}

        # 按总计数排序
        sorted_nouns = sorted(noun_attrs.items(), key=lambda x: x[1]['count'], reverse=True)
        sorted_adjs = sorted(adj_attrs.items(), key=lambda x: x[1]['count'], reverse=True)

        # 计算所需数量
        nouns_needed = int(max_attrs_per_brand * target_noun_ratio)
        adjs_needed = int(max_attrs_per_brand * target_adj_ratio)

        # 调整数量以确保总数不超过max_attrs_per_brand
        if nouns_needed + adjs_needed > max_attrs_per_brand:
            nouns_needed = max_attrs_per_brand - adjs_needed

        # 如果某类词不足，调整另一类词的数量
        if len(sorted_nouns) < nouns_needed:
            nouns_actual = len(sorted_nouns)
            adjs_needed = min(len(sorted_adjs), max_attrs_per_brand - nouns_actual)
            nouns_needed = nouns_actual

        if len(sorted_adjs) < adjs_needed:
            adjs_actual = len(sorted_adjs)
            nouns_needed = min(len(sorted_nouns), max_attrs_per_brand - adjs_actual)
            adjs_needed = adjs_actual

        # 选择名词和形容词
        selected_nouns = dict(sorted_nouns[:nouns_needed])
        selected_adjs = dict(sorted_adjs[:adjs_needed])

        # 合并选择的属性
        selected_attrs = {**selected_nouns, **selected_adjs}

        # 计算实际比例
        actual_nouns = len(selected_nouns)
        total_selected = len(selected_attrs)
        actual_noun_ratio = actual_nouns / total_selected if total_selected > 0 else 0
        actual_adj_ratio = 1 - actual_noun_ratio

        print(f"  品牌 '{brand_name}' 选择了 {total_selected} 个属性: "
              f"名词 {actual_nouns}个 ({actual_noun_ratio:.2%}), "
              f"形容词 {total_selected - actual_nouns}个 ({actual_adj_ratio:.2%})")

        # 创建当前品牌的属性数据列表
        brand_data = []
        for attr, info in selected_attrs.items():
            # 为每个属性创建一行数据
            row_data = {
                '品牌': brand_name,
                '属性': attr,
                '词性': '名词' if info.get('pos') == 'NN' else '形容词' if info.get('pos') == 'J' else '未知',
                '总计数': info['count'],
                '情感类型': 'positive' if info['sentiment_score'] >=0.8 else 'negative',
                '情感得分': info['sentiment_score'],
                '正面计数': info['positive_count'],
                '负面计数': info['negative_count'],
                '中性计数': info['neutral_count'],
                '情感强度': info['strength'],
                '中心度': info.get('nwcd', 0),
                '文档频率': info.get('doc_freq', 0),
                'TF-IDF得分': info.get('tf_idf', 0),
                '加权正面': info['weighted_positive'],
                '加权负面': info['weighted_negative'],
                '加权中性': info['weighted_neutral']
            }
            brand_data.append(row_data)

        # 将这个品牌的数据追加到主DataFrame
        brand_df = pd.DataFrame(brand_data)
        df = pd.concat([df, brand_df], ignore_index=True)

    # 保存样式化的Excel
    try:
        # 创建一个Excel writer对象
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

        # 将数据写入Excel
        df.to_excel(writer, index=False, sheet_name='品牌属性分析')

        # 获取xlsxwriter对象
        workbook = writer.book
        worksheet = writer.sheets['品牌属性分析']

        # 添加一些格式
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })

        # 添加词性特殊格式
        noun_format = workbook.add_format({
            'bg_color': '#E6F2FF',  # 浅蓝色背景用于名词
            'border': 1
        })

        adj_format = workbook.add_format({
            'bg_color': '#FFEBCC',  # 浅橙色背景用于形容词
            'border': 1
        })

        # 格式化整个表格
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            # 设置列宽
            if value == '品牌' or value == '属性':
                worksheet.set_column(col_num, col_num, 15)
            else:
                worksheet.set_column(col_num, col_num, 12)

        # 为名词和形容词设置不同的行背景色
        for row_num, row_data in enumerate(df.values, start=1):
            if row_data[2] == '名词':  # 词性列索引为2
                for col_num, value in enumerate(row_data):
                    worksheet.write(row_num, col_num, value, noun_format)
            elif row_data[2] == '形容词':
                for col_num, value in enumerate(row_data):
                    worksheet.write(row_num, col_num, value, adj_format)

        # 关闭writer
        writer.close()
        print(f"品牌属性信息已导出到: {output_file} (已增强格式)")
    except Exception as e:
        print(f"保存增强版Excel出错: {e}")
        # 如果增强版导出失败，尝试普通导出
        df.to_excel(output_file, index=False)
        print(f"品牌属性信息已使用标准格式导出到: {output_file}")

    # 返回统计信息
    brands_count = len(df['品牌'].unique())
    attributes_count = len(df)
    positive_count = len(df[df['情感类型'] == 'positive'])
    negative_count = len(df[df['情感类型'] == 'negative'])
    neutral_count = len(df[df['情感类型'] == 'neutral'])
    noun_count = len(df[df['词性'] == '名词'])
    adj_count = len(df[df['词性'] == '形容词'])

    print(f"导出统计: {brands_count}个品牌, {attributes_count}个属性")
    print(f"其中: 正面属性 {positive_count}个, 负面属性 {negative_count}个, 中性属性 {neutral_count}个")
    print(f"词性分布: 名词 {noun_count}个 ({noun_count / attributes_count:.2%}), "
          f"形容词 {adj_count}个 ({adj_count / attributes_count:.2%})")

    return df

# 计算品牌属性的唯一性
def visualize_brand_uniqueness(brand1, brand2, brand_attributes, save_path=None):
    """
    生成品牌属性唯一性表格可视化，类似论文中的Table 4
    :param brand1: 第一个品牌名称
    :param brand2: 第二个品牌名称
    :param brand_attributes: 品牌属性字典
    :param save_path: 保存路径
    :return: None
    """
    print(f"创建 {brand1} 和 {brand2} 的唯一性表格可视化...")

    # 获取可用的中文字体
    chinese_font = setup_fonts()

    # 计算唯一性
    def calc_uniqueness(attr, brand1, brand2):
        nwcd_1 = brand_attributes[brand1].get(attr, {}).get('nwcd', 0)
        nwcd_2 = brand_attributes[brand2].get(attr, {}).get('nwcd', 0)
        max_nwcd = max(nwcd_1, nwcd_2)
        if max_nwcd == 0:
            return 0
        return (nwcd_1 - nwcd_2) / max_nwcd

    # 获取两个品牌的前10个属性（按NWCD排序）
    top_attrs_1 = sorted(
        [(attr, info) for attr, info in brand_attributes[brand1].items()],
        key=lambda x: x[1]['nwcd'],
        reverse=True
    )[:10]

    top_attrs_2 = sorted(
        [(attr, info) for attr, info in brand_attributes[brand2].items()],
        key=lambda x: x[1]['nwcd'],
        reverse=True
    )[:10]

    # 创建表格数据
    data_brand1 = []
    for attr, info in top_attrs_1:
        data_brand1.append({
            'brand_association': attr,
            'favorability': 'pro' if info['sentiment'] == 'positive' else 'con',
            'uniqueness': calc_uniqueness(attr, brand1, brand2)
        })

    data_brand2 = []
    for attr, info in top_attrs_2:
        data_brand2.append({
            'brand_association': attr,
            'favorability': 'pro' if info['sentiment'] == 'positive' else 'con',
            'uniqueness': calc_uniqueness(attr, brand2, brand1)
        })

    # 创建图形
    plt.rcParams['font.family'] = chinese_font  # 设置全局字体
    fig, ax = plt.subplots(figsize=(14, 10))  # 增加图表大小

    # 移除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)

    # 设置标题
    ax.set_title('Uniqueness of Top Ten Associations Based on Normalized,\nWeighted Degree Centrality',
                 fontsize=18, fontweight='bold', pad=20, fontfamily=chinese_font)

    # 去除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])

    # 创建表格的列名
    column_labels = ['Brand\nassociation', 'Favorability', 'Uniqueness']

    # 创建表格单元格数据
    table_data = []

    # 添加品牌1和品牌2的标题行
    table_data.append([brand1, '', ''])

    # 添加品牌1数据
    for row in data_brand1:
        table_data.append([
            row['brand_association'],
            row['favorability'],
            f"{row['uniqueness']:.3f}"
        ])

    # 添加品牌2标题行（空行分隔）
    table_data.append(['', '', ''])
    table_data.append([brand2, '', ''])

    # 添加品牌2数据
    for row in data_brand2:
        table_data.append([
            row['brand_association'],
            row['favorability'],
            f"{row['uniqueness']:.3f}"
        ])

    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        colWidths=[0.5, 0.25, 0.25],  # 增加第一列宽度以容纳中文
        loc='center',
        cellLoc='center'
    )

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)  # 增加行高以避免文本重叠

    # 设置标题行的样式
    for i, key in enumerate(table._cells):
        cell = table._cells[key]

        # 设置所有单元格的字体为中文字体
        cell.get_text().set_fontfamily(chinese_font)

        # 品牌标题行
        if key[0] == 0 or key[0] == 1 or key[0] == len(data_brand1) + 3:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')

        # 列标题行
        if key[0] == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d9d9d9')

        # 分隔行
        if key[0] == len(data_brand1) + 2:
            cell.set_facecolor('#f2f2f2')

    # 添加底部注释
    note_text = "Notes: Uniqueness measure ranges between -1 and +1. A value of -1 indicates that the brand association is\nunique for the focal brand. A value of zero indicates that the two brands share the brand association, and a\nvalue of +1 indicates that the brand association is unique for the competing brand."
    ax.text(0.5, 0.02, note_text,
            ha='center', va='center',
            fontsize=10, fontstyle='italic',
            transform=ax.transAxes,
            fontfamily=chinese_font)

    plt.tight_layout(pad=2.0)  # 增加内边距

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  唯一性表格已保存到: {save_path}")

    plt.show()

# 修改后的主函数
def main():
    # 1. 加载预处理后的数据，并按品牌分组
    processed_df = load_processed_data_by_brand('processed_reviews.xlsx', brand)
    print(f"加载了 {len(processed_df)} 条评论记录")

    # 2. 提取品牌属性 (调整min_frequency参数可以控制保留词的数量)
    brand_attributes = extract_brand_attributes(processed_df, min_frequency=3, min_word_length=2)

    # 3. 计算属性中心度
    brand_attributes = calculate_brand_centrality(brand_attributes, processed_df)

    # 4. 识别属性间关联
    connections = identify_brand_attribute_connections(processed_df, brand_attributes)

    # 5. 为每个品牌构建并可视化网络
    focus_brands = ['比亚迪', '特斯拉', '吉利', '理想', '广汽埃安', '问界', '零跑', '蔚来', '小鹏', '小米']  # 这里可以关注特定的品牌进行可视化
    for brand_name in focus_brands:
        if brand_name in brand_attributes and len(brand_attributes[brand_name]) > 5:  # 只处理有足够属性的品牌
            # 先应用离群值检测，避免孤立的节点
            attributes = brand_attributes[brand_name]

            # 检测离群点
            if len(attributes) > 0:
                # 计算属性连接数
                attr_connections = {}
                for attr in attributes:
                    attr_connections[attr] = 0
                    for other_attr in attributes:
                        if attr != other_attr and attr in connections[brand_name] and other_attr in \
                                connections[brand_name][attr]:
                            if connections[brand_name][attr][other_attr] > 0:
                                attr_connections[attr] += 1

                # 找出连接数为0或者很少的离群属性
                isolated_attrs = [attr for attr, conn_count in attr_connections.items()
                                  if conn_count <= 1]  # 连接数小于等于1的视为离群点

                print(f"  发现 {len(isolated_attrs)} 个离群属性，从图中移除")

                # 从属性字典中移除离群点
                for attr in isolated_attrs:
                    if attr in attributes:
                        del attributes[attr]

            # 使用增强的网络构建和可视化函数
            G = build_brand_network(brand_name, brand_attributes, connections,
                                             min_edge_weight=1, top_attributes=40)

            title = f"{brand_name} 品牌评论属性关联网络"
            save_path = f"{brand_name}_brand_network_enhanced.png"
            visualize_brand_network(G, title, save_path)

            # 输出该品牌的主要属性
            print(f"\n品牌 {brand_name} 的主要属性:")
            print("正面属性:")
            sorted_positive = sorted([i for i in brand_attributes[brand_name].items()
                                      if i[1]['sentiment'] == 'positive'],
                                     key=lambda x: x[1]['nwcd'], reverse=True)

            for attr, info in sorted_positive[:10]:  # 显示前10个正面属性
                print(f"  - {attr}: 中心度: {info['nwcd']:.4f}, 计数: {info['count']}")

            print("\n负面属性:")
            sorted_negative = sorted([i for i in brand_attributes[brand_name].items()
                                      if i[1]['sentiment'] == 'negative'],
                                     key=lambda x: x[1]['nwcd'], reverse=True)

            for attr, info in sorted_negative[:10]:  # 显示前10个负面属性
                print(f"  - {attr}: 中心度: {info['nwcd']:.4f}, 计数: {info['count']}")

    # 6. 创建品牌唯一性表格可视化（对比小米和特斯拉）
    if '小米' in brand_attributes and '特斯拉' in brand_attributes:
        uniqueness_save_path = "uniqueness_table_小米_特斯拉.png"
        visualize_brand_uniqueness('小米', '特斯拉', brand_attributes, uniqueness_save_path)

    # 6. 创建品牌唯一性表格可视化（对比小米和特斯拉）
    if '理想' in brand_attributes and '蔚来' in brand_attributes:
        uniqueness_save_path = "uniqueness_table_理想_蔚来.png"
        visualize_brand_uniqueness('理想', '蔚来', brand_attributes, uniqueness_save_path)
    
        
    # 导出所有品牌属性到Excel
    export_df = export_brand_attributes_to_excel(brand_attributes)
    print("\n分析完成!")

# 运行主函数
if __name__ == "__main__":
   main()

# 原始数据读取
'''

# 按车型分组并处理评论
def process_reviews_by_model():
    """
    按车型处理评论数据
    @return {DataFrame} 处理后的数据框，包含车型和对应的所有评论
    """
    processed_data = []
    
    # 按car_model分组
    for model in df['car_model'].unique():
        model_df = df[df['car_model'] == model]
        
        # 收集该车型的所有评论
        all_reviews = []
        for _, row in model_df.iterrows():
            # 获取该行的所有评论
            reviews = [row[col] for col in review_columns if pd.notna(row[col])]
            all_reviews.extend(reviews)
            
        processed_data.append({
            'model': model,
            'review_text': all_reviews
        })
    
    return pd.DataFrame(processed_data)


input_file = "vehiclenew.xlsx"

print(f"正在读取文件: {input_file}")

# 读取Excel文件，跳过第一行（标题行），从第二行开始
df = pd.read_excel(input_file, skiprows=0)

review_columns = [
    'most_sat', 'least_sat', 'space_desc', 'drive_exp_desc', 
    'range_desc', 'exterior_desc', 'interior_desc', 
    'cost_perf_desc', 'allocation_desc'
]

# 获取处理后的数据
example_data = process_reviews_by_model()

# 查看前20行完整数据
print(example_data.head(20))



# 数据预处理
# processed_df = preprocess_reviews(example_data)

# 加载数据
processed_df = load_processed_data('processed_reviews.xlsx')
print(f"加载了 {len(processed_df)} 条评论记录")

print(processed_df)
# 处理数据


# 将结果保存为Excel文件

test_results = processed_df
test_results.to_excel('processed_reviews.xlsx', index=False)
print(f"数据已保存到 'processed_reviews.xlsx'")

'''


