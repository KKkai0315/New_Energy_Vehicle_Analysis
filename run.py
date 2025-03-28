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
from tqdm import tqdm

warnings.filterwarnings('ignore')


# 创建AcsClient实例
client = AcsClient(
    access_key_id,
    access_key_secret,
    "cn-hangzhou"
)

# 品牌映射字典
brand_mapping = dict({
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


# 检查当前可用字体
def setup_fonts():
    chinese_fonts = ['STSong', 'PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'WenQuanYi Micro Hei']

    available_font = None
    for font in chinese_fonts:
        if any(font.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            available_font = font
            break

    if not available_font:
        available_font = 'Arial Unicode MS'

    print(f"使用字体: {available_font}")
    return available_font


# 设置停用词表
def get_stopwords():
    """
    获取停用词表，这些词不会被视为有意义的品牌联想
    """
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
        "半年","后方","警示","理念","评论","差别","记录","地下","商场","双手","相关","流行","有关"

    ])
    return stopwords


# 数据预处理
def preprocess_reviews(reviews_df):
    """
    按照人类联想记忆模型框架处理用户评论数据
    :param reviews_df: 包含用户评论的DataFrame
    :return: 处理后的数据
    """
    processed_data = []
    error_count = 0
    total_reviews = 0

    for idx, row in reviews_df.iterrows():
        try:
            model = row['model']
            review_texts = row['review_text']  # 评论文本列表
            # 远程调用服务器上的模型
            for text in review_texts:
                total_reviews += 1
                try:
                    # 分词处理
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

                    words = []
                    result_data = json.loads(resp_obj['Data'])
                    for word_item in result_data['result']:
                        # 过滤基本词
                        if not word_item['tags'] or len(word_item['tags']) > 1 or word_item['tags'][0] != "基本词-中文":
                            words.append(word_item['word'])

                    # 情感分析
                    sentiment_request = CommonRequest()
                    sentiment_request.set_domain('alinlp.cn-hangzhou.aliyuncs.com')
                    sentiment_request.set_version('2020-06-29')
                    sentiment_request.set_action_name('GetSaChGeneral')
                    sentiment_request.add_query_param('ServiceCode', 'alinlp')
                    sentiment_request.add_query_param('Text', text)

                    sentiment_response = client.do_action_with_exception(sentiment_request)
                    sentiment_obj = json.loads(sentiment_response)
                    sentiment_data = json.loads(sentiment_obj['Data'])

                    # 获取情感标签和概率
                    sentiment = sentiment_data['result']['sentiment']
                    positive_prob = sentiment_data['result']['positive_prob']
                    negative_prob = sentiment_data['result']['negative_prob']
                    neutral_prob = sentiment_data['result']['neutral_prob']

                    # 确定评论部分（pros或cons）
                    review_section = 'pros' if positive_prob > negative_prob else 'cons'

                    processed_data.append({
                        'model': model,
                        'words': words,
                        'sentiment': sentiment,
                        'text': text,
                        'review_section': review_section,
                        'positive_prob': positive_prob,
                        'negative_prob': negative_prob,
                        'neutral_prob': neutral_prob
                    })

                except Exception as e:
                    error_count += 1
                    print(f"处理单条评论错误(行 {idx}): {e}")
                    continue
        except Exception as e:
            print(f"处理数据行错误(行 {idx}): {e}")
            continue

    print(f"处理完成: 总共处理 {total_reviews} 条评论，跳过 {error_count} 条错误")
    return pd.DataFrame(processed_data)


def identify_part_of_speech(word_list, client, required_ratio={'NN': 0.6, 'JJ': 0.4}):
    """
    使用阿里云API识别词性并按照指定比例筛选属性
    :param word_list: 待检查词性的单词列表
    :param client: 阿里云客户端
    :param required_ratio: 所需的词性比例，如 {'NN': 0.6, 'JJ': 0.4} 表示60%名词，40%形容词
    :return: 词性字典和按比例筛选后的词列表
    """
    print(f"正在识别 {len(word_list)} 个词的词性...")

    batch_size = 100  # 每次处理100个词
    pos_dict = {}

    for i in range(0, len(word_list), batch_size):
        batch_words = word_list[i:min(i + batch_size, len(word_list))]
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
                if 'result' in result_data:
                    for item in result_data['result']:
                        word = item['word']
                        pos = item['pos']  # 词性代码: NN-名词, JJ-形容词, 等

                        # 只关注名词和形容词
                        if pos in ['NN', 'JJ']:
                            pos_dict[word] = pos
        except Exception as e:
            print(f"词性识别API错误: {e}")
            continue

    noun_count = sum(1 for pos in pos_dict.values() if pos == 'NN')
    adj_count = sum(1 for pos in pos_dict.values() if pos == 'JJ')
    print(f"词性识别结果: 名词 {noun_count} 个, 形容词 {adj_count} 个")

    nouns = [word for word, pos in pos_dict.items() if pos == 'NN']
    adjs = [word for word, pos in pos_dict.items() if pos == 'JJ']

    total_needed = len(word_list)
    nouns_needed = int(total_needed * required_ratio['NN'])
    adjs_needed = int(total_needed * required_ratio['JJ'])

    if nouns_needed + adjs_needed < total_needed:
        nouns_needed += (total_needed - (nouns_needed + adjs_needed))

    if len(nouns) < nouns_needed:
        nouns_actual = len(nouns)
        adjs_needed += (nouns_needed - nouns_actual)
        nouns_needed = nouns_actual

    if len(adjs) < adjs_needed:
        adjs_actual = len(adjs)
        nouns_needed += (adjs_needed - adjs_actual)
        adjs_needed = adjs_actual

    selected_nouns = nouns[:nouns_needed] if nouns_needed <= len(nouns) else nouns
    selected_adjs = adjs[:adjs_needed] if adjs_needed <= len(adjs) else adjs

    selected_words = selected_nouns + selected_adjs

    actual_ratio = {
        'NN': len(selected_nouns) / max(1, len(selected_words)),
        'JJ': len(selected_adjs) / max(1, len(selected_words))
    }

    print(
        f"词性筛选后: 名词 {len(selected_nouns)}个 ({actual_ratio['NN']:.2%}), 形容词 {len(selected_adjs)}个 ({actual_ratio['JJ']:.2%})")

    return pos_dict, selected_words


# 加载预处理数据
def load_processed_data_by_brand(file_path='processed_reviews.xlsx', brand_mapping=None):
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
    if brand_mapping:
        for brand, models in brand_mapping.items():
            for model in models:
                model_to_brand[model] = brand

        # 添加品牌列
        processed_df['brand'] = processed_df['model'].map(model_to_brand)

        brand_counts = processed_df['brand'].value_counts()
        print("\n===== 各品牌数据量统计 =====")
        for brand, count in brand_counts.items():
            print(f"品牌: {brand} - 数据量: {count}条")
        print("==========================\n")

    # 将字符串形式的词列表转换回Python列表
    if 'words' in processed_df.columns:
        if processed_df.shape[0] > 0:  # 确保有数据
            first_words = processed_df['words'].iloc[0]


            if not isinstance(first_words, list):

                def parse_words(words_str):
                    if pd.isna(words_str):
                        return []

                    if isinstance(words_str, str):

                        if words_str.startswith('[') and words_str.endswith(']'):
                            try:
                                return eval(words_str)
                            except:

                                clean_str = words_str.strip('[]').replace("'", "").replace('"', '')
                                return [w.strip() for w in clean_str.split(',')]
                        else:

                            return [words_str]
                    elif isinstance(words_str, list):
                        return words_str
                    else:
                        return []

                processed_df['words'] = processed_df['words'].apply(parse_words)
                print("已将字符串形式的words列表转换为Python列表")


        if processed_df.shape[0] > 0:
            print(f"第一行words示例: {processed_df['words'].iloc[0][:5]}...")


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



def extract_brand_attributes(processed_df, min_frequency=2, min_word_length=2):
    """
    提取品牌关联属性，基于人类联想记忆模型
    :param processed_df: 处理后的数据
    :param min_frequency: 最小出现频率
    :param min_word_length: 最小词长度
    :return: 品牌属性字典
    """
    print("正在提取品牌属性（按词性比例筛选）...")
    brand_attributes = {}
    stopwords = get_stopwords()

    # 针对选定的品牌进行分析
    selected_brands = ['比亚迪', '特斯拉', '吉利', '理想', '广汽埃安', '问界', '零跑', '蔚来', '小鹏', '小米']

    for brand in processed_df['brand'].unique():
        if brand not in selected_brands:
            continue

        brand_df = processed_df[processed_df['brand'] == brand]

        # 收集所有词及其情感
        word_sentiments = {}
        for _, row in brand_df.iterrows():


            positive_prob = row['positive_prob']
            sentiment_key = 'positive' if positive_prob >= 0.88 else 'negative'

            weight = row['positive_prob'] if sentiment_key == 'positive' else row['negative_prob']

            for word in row['words']:
                # 过滤停用词和短词
                if word in stopwords or len(word) < min_word_length:
                    continue

                if word not in word_sentiments:
                    word_sentiments[word] = {'positive': 0, 'negative': 0, 'neutral': 0,
                                             'weighted_positive': 0, 'weighted_negative': 0, 'weighted_neutral': 0}
                word_sentiments[word][sentiment_key] += 1
                word_sentiments[word][f'weighted_{sentiment_key}'] += weight

        # 过滤低频词，对负面属性使用更低的阈值
        filtered_attributes = {}
        for word, counts in word_sentiments.items():
            total = counts['positive'] + counts['negative'] + counts['neutral']

            # 判断主要情感
            if counts['positive'] >= counts['negative'] and counts['positive'] >= counts['neutral']:
                main_sentiment = 'positive'
                strength = counts['positive'] / total
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
                # 计算情感得分 ，-1～1
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

        # 词性分析并按比例筛选
        attribute_words = list(filtered_attributes.keys())
        if attribute_words:

            pos_dict, selected_words = identify_part_of_speech(
                attribute_words,
                client,
                required_ratio={'NN': 0.6, 'JJ': 0.4}
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


#计算品牌属性中心度
def calculate_brand_centrality(brand_attributes, processed_df):
    """
    基于论文方法计算品牌属性的中心度 - 使用向量化操作提高速度
    :param brand_attributes: 品牌属性字典
    :param processed_df: 所有评论数据
    :return: 带中心度的品牌属性字典
    """
    print("正在计算品牌属性中心度...")

    for brand in tqdm(brand_attributes.keys(), desc="计算各品牌属性中心度"):
        attributes = brand_attributes[brand]
        if not attributes:
            continue

        brand_df = processed_df[processed_df['brand'] == brand]
        total_reviews = len(brand_df)

        if total_reviews == 0:
            print(f"警告: 品牌 '{brand}' 没有评论数据")
            continue

        # 预计算每个产品的评论数量
        product_review_counts = brand_df['model'].value_counts().to_dict()

        # 获取所有产品
        all_products = list(product_review_counts.keys())

        # 计算每个属性的NWCD
        # # NWCDᵢ = Σₚ [rₚ/|R| · xᵢₚ/|Iₚ|] · 100
        for attr, info in tqdm(attributes.items(), desc=f"计算 '{brand}' 的属性中心度", leave=False):
            nwcd = 0
            doc_count = 0

            # 使用DataFrame操作替代循环
            for product in all_products:
                product_df = brand_df[brand_df['model'] == product]
                r_p = len(product_df)  # 产品评论数

                # 产品中出现的次数 - 使用向量化操作
                x_ip = sum(1 for _, row in product_df.iterrows() if attr in row['words'])

                # 统计文档频率
                doc_count += x_ip

                # 产品中所有属性提及的总数
                total_mentions = sum(len(row['words']) for _, row in product_df.iterrows())

                if total_mentions > 0:
                    # 产品对属性中心度的贡献
                    contribution = (r_p / total_reviews) * (x_ip / total_mentions)
                    nwcd += contribution

            # 获得百分比并保存
            nwcd *= 100
            attributes[attr]['nwcd'] = nwcd
            attributes[attr]['doc_freq'] = doc_count

            # 计算TF-IDF得分
            tf = info['count']  # 词频
            idf = np.log(total_reviews / (1 + doc_count))  # 逆文档频率
            tf_idf = tf * idf
            attributes[attr]['tf_idf'] = tf_idf

    # 检测和移除离群值
    for brand, attributes in brand_attributes.items():
        if len(attributes) > 0:
            # 计算NWCD的四分位数
            nwcd_values = [info['nwcd'] for info in attributes.values()]
            q1 = np.percentile(nwcd_values, 25)
            q3 = np.percentile(nwcd_values, 75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr

            # 找到离群值
            outliers = [attr for attr, info in attributes.items() if info['nwcd'] > upper_bound]
            print(f"  品牌 '{brand}' 检测到 {len(outliers)} 个中心度离群值")

            # 移除离群值
            for outlier in outliers:
                del attributes[outlier]

    return brand_attributes


# 识别品牌属性间关联
def identify_brand_attribute_connections(processed_df, brand_attributes):
    """
    :param processed_df: 处理后的数据
    :param brand_attributes: 品牌属性字典
    :return: 关联图
    """
    print("正在识别品牌属性间关联...")
    connections = {}

    for brand in tqdm(brand_attributes.keys(), desc="处理品牌间关联"):
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

        # 共现次数 - 使用更高效的向量化操作
        for _, row in tqdm(brand_df.iterrows(), desc=f"处理 '{brand}' 的评论共现", leave=False, total=len(brand_df)):
            words_in_review = set(word for word in row['words'] if word in attributes)
            for attr1 in words_in_review:
                for attr2 in words_in_review:
                    if attr1 != attr2:
                        connections[brand][attr1][attr2] += 1

        print(f"  品牌 '{brand}' 已计算属性关联")

    return connections


def build_brand_network(brand, brand_attributes, connections, min_edge_weight=1, top_attributes=100):
    """
    基于人类联想记忆模型构建品牌关联网络，表示品牌形象
    :param brand: 品牌名
    :param brand_attributes: 品牌属性字典
    :param connections: 属性间连接
    :param min_edge_weight: 最小边权重
    :param top_attributes: 显示前N个最重要的属性
    :return: NetworkX图对象
    """
    print(f"正在构建 '{brand}' 的网络图 (基于人类联想记忆模型)...")
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

    # 合理的正负面属性比例
    target_negative_ratio = 0.3  # 目标负面属性比例
    min_negative_count = min(len(negative_attrs), max(5, int(top_attributes * target_negative_ratio)))
    target_neutral_ratio = 0.1  # 目标中性属性比例
    min_neutral_count = min(len(neutral_attrs), max(3, int(top_attributes * target_neutral_ratio)))

    # 剩余槽位分配给正面属性
    remaining_slots = top_attributes - min_negative_count - min_neutral_count

    # 确保词性比例符合要求: 60%名词, 40%形容词
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
                if connections[brand][attr1][attr2] >= min_edge_weight:
                    attr_connections[attr1] += 1

            # 找出最大和最小连接数，用于归一化
        max_connections = max(attr_connections.values()) if attr_connections else 1
        min_connections = min(attr_connections.values()) if attr_connections else 0

        # 添加品牌节点作为中心节点
        G.add_node(brand, type='brand', size=50, sentiment='neutral', sentiment_score=0)

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
            connection_weight = 0.4  # 连接数权重

            # 对词频和中心度进行归一化
            normalized_count = np.log1p(info['count'])  # 对数变换减少极端值影响
            normalized_nwcd = nwcd * 100  # 缩放中心度

            # 计算最终大小 - 增加连接数的影响
            size = (count_weight * normalized_count +
                    nwcd_weight * normalized_nwcd +
                    connection_weight * (norm_connections * 10 + 1)) * 3.0

            # 确保最小可见大小
            size = max(16, size)

            # 情感得分用于渐变色彩
            sentiment_score = info['sentiment_score']

            # 将属性节点添加到图中
            G.add_node(attr,
                       type='attribute',
                       size=size,
                       sentiment=info['sentiment'],
                       sentiment_score=sentiment_score,
                       count=info['count'],
                       nwcd=nwcd,
                       pos=info.get('pos', 'unknown'),
                       connection_count=connection_count)

            # 将属性与品牌相连
            G.add_edge(brand, attr, weight=nwcd)

        # 添加属性之间的边 - 基于共现次数
        for attr1 in selected_attrs:
            for attr2 in selected_attrs:
                if attr1 != attr2 and attr1 in connections[brand] and attr2 in connections[brand][attr1]:
                    weight = connections[brand][attr1][attr2]
                    if weight >= min_edge_weight:
                        G.add_edge(attr1, attr2, weight=weight)

        negative_count = sum(1 for node in G.nodes() if G.nodes[node].get('sentiment') == 'negative')
        print(f"  添加了 {len(G.nodes())} 个节点 (其中负面属性 {negative_count} 个) 和 {len(G.edges())} 条边")

        return G


# 可视化网络图
def visualize_brand_network(G, title, save_path=None):
    """
    :param G: NetworkX图对象
    :param title: 图标题
    :param save_path: 保存路径
    """
    print(f"正在可视化 '{title}' 网络图...")

    # 获取可用的中文字体
    chinese_font = setup_fonts()

    # 设置绘图样式
    plt.style.use('seaborn-v0_8-whitegrid')

    # 创建图形
    fig, ax = plt.subplots(figsize=(18, 15), dpi=100)

    # 设置背景颜色
    bg_color = '#e9eff5'  # 浅蓝灰色
    fig.patch.set_facecolor(bg_color)  # 图形背景
    ax.set_facecolor('#ffffff')  # 保持图表区域为白色

    # 移除孤立节点
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        print(f"  移除 {len(isolated_nodes)} 个孤立节点")
        G.remove_nodes_from(isolated_nodes)

    # 使用更优化的布局
    pos = nx.kamada_kawai_layout(G)

    # 准备节点样式
    node_sizes = []
    node_colors = []
    node_borders = []
    border_widths = []
    node_labels = {}

    # 创建更鲜艳的颜色方案
    colors = [(0.9, 0.1, 0.1), (1, 1, 1), (0.1, 0.3, 0.9)]  # 红白蓝渐变
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    # 计算节点的连接数
    node_connection_counts = {}
    for node in G.nodes():
        node_connection_counts[node] = len(list(G.neighbors(node)))

    # 处理品牌和属性节点
    brand_nodes = []
    attr_nodes = []

    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'attribute')
        if node_type == 'brand':
            brand_nodes.append(node)
        else:
            attr_nodes.append(node)

    for node in G.nodes():
        # 获取节点属性
        sentiment = G.nodes[node].get('sentiment', 'neutral')
        sentiment_score = G.nodes[node].get('sentiment_score', 0)
        node_size = G.nodes[node].get('size', 15)
        node_type = G.nodes[node].get('type', 'attribute')
        connection_count = node_connection_counts[node]

        # 区分品牌和属性节点
        if node_type == 'brand':
            # 品牌节点使用较大尺寸
            node_sizes.append(node_size * 6)
            node_colors.append('#3498db')  # 品牌节点使用蓝色
            node_borders.append('#2980b9')
            border_widths.append(2.0)
            # 为品牌节点创建更大的标签
            node_labels[node] = node
        else:
            # 属性节点大小基于中心度和连接数
            node_sizes.append(node_size)

            # 颜色基于情感倾向
            if sentiment == 'positive':
                # 正向 - 蓝色调
                blue_intensity = min(1.0, 0.4 + 0.6 * sentiment_score)
                rgba_color = (0.0, 0.2 * (1 - sentiment_score), blue_intensity, 0.95)
            elif sentiment == 'negative':
                # 负向 - 红色调
                red_intensity = min(1.0, 0.4 + 0.6 * abs(sentiment_score))
                rgba_color = (red_intensity, 0.0, 0.1 * (1 - abs(sentiment_score)), 0.95)
            else:  # 中性
                rgba_color = (0.7, 0.7, 0.7, 0.85)

            node_colors.append(rgba_color)

            # 边框颜色
            if sentiment_score < -0.4:  # 负面
                border_color = '#8a2a2a'  # 深红色
            elif sentiment_score > 0.4:  # 正面
                border_color = '#2a4c8a'  # 深蓝色
            else:  # 中性
                border_color = '#5c656d'  # 灰色

            node_borders.append(border_color)
            border_widths.append(1.5)

            # 为属性节点创建标签
            node_labels[node] = node

    # 准备边的样式
    edge_widths = []
    edge_alphas = []

    for u, v in G.edges():
        # 获取边权重
        weight = G.edges[u, v].get('weight', 1.0)

        # 对于和品牌相连的边
        if u in brand_nodes or v in brand_nodes:
            # 使边更粗，表示与品牌的关联强度
            edge_width = weight * 0.1 + 0.5  # 适当缩放
            edge_alpha = 0.7
        else:
            # 属性间的边
            edge_width = max(0.5, min(3.0, weight * 0.05))
            edge_alpha = 0.5

        edge_widths.append(edge_width)
        edge_alphas.append(edge_alpha)

    # 绘制边 - 使用黑色边
    edges = nx.draw_networkx_edges(G, pos,
                                   width=edge_widths,
                                   alpha=edge_alphas,
                                   edge_color='#000000',
                                   connectionstyle='arc3,rad=0.05',  # 稍微弯曲的边
                                   ax=ax)

    # 绘制属性节点
    nodes = nx.draw_networkx_nodes(G, pos,
                                   nodelist=attr_nodes,
                                   node_size=node_sizes[len(brand_nodes):],
                                   node_color=node_colors[len(brand_nodes):],
                                   edgecolors=node_borders[len(brand_nodes):],
                                   linewidths=border_widths[len(brand_nodes):],
                                   alpha=0.92,
                                   ax=ax)

    # 绘制品牌节点
    brand_nodes_drawn = nx.draw_networkx_nodes(G, pos,
                                               nodelist=brand_nodes,
                                               node_size=node_sizes[:len(brand_nodes)],
                                               node_color=node_colors[:len(brand_nodes)],
                                               edgecolors=node_borders[:len(brand_nodes)],
                                               linewidths=border_widths[:len(brand_nodes)],
                                               alpha=1.0,
                                               ax=ax)

    # 绘制标签
    for node in G.nodes():
        x, y = pos[node]
        node_type = G.nodes[node].get('type', 'attribute')

        if node_type == 'brand':
            # 品牌标签更大更粗
            plt.text(x, y, node_labels[node],
                     fontsize=16,
                     fontfamily=chinese_font,
                     ha='center', va='center',
                     color='white',
                     weight='bold')
        else:
            # 属性标签
            sentiment = G.nodes[node].get('sentiment', 'neutral')

            # 计算字体大小
            base_size = 9
            connection_count = G.nodes[node].get('connection_count', 0)
            max_connections = max(node_connection_counts.values()) if node_connection_counts else 1
            conn_factor = 1 + 0.3 * (connection_count / max_connections)
            font_size = min(14, max(9, base_size * conn_factor))

            # 文本颜色
            if sentiment == 'positive' or sentiment == 'negative':
                text_color = 'white'  # 对正负面情感使用白色文本
            else:
                text_color = 'black'  # 中性情况用黑色

            plt.text(x, y, node_labels[node],
                     fontsize=font_size,
                     fontfamily=chinese_font,
                     ha='center', va='center',
                     color=text_color,
                     weight='bold' if sentiment in ['positive', 'negative'] else 'normal')

    # 添加颜色条 - 表示情感分数
    norm = plt.Normalize(-1, 1)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical',
                        pad=0.01, fraction=0.02,
                        location='right')
    cbar.set_label('情感分数', fontfamily=chinese_font, fontsize=12, color='#343a40')
    cbar.ax.tick_params(colors='#343a40', labelsize=10)

    # 设置标题
    plt.title(title, fontsize=22, fontfamily=chinese_font, pad=20,
              color='#343a40', fontweight='normal')

    # 设置精确的边界，减少空白区域
    plt.tight_layout(pad=3.0)
    ax.set_axis_off()

    # 添加边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_color('#e9ecef')
    ax.spines['right'].set_color('#e9ecef')
    ax.spines['bottom'].set_color('#e9ecef')
    ax.spines['left'].set_color('#e9ecef')

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"  图像已保存到: {save_path}")

    plt.show()


# 计算品牌属性的唯一性
def calculate_uniqueness(brand_attributes, compare_brand_attributes):
    """
    计算品牌属性的唯一性
    :param brand_attributes: 焦点品牌的属性
    :param compare_brand_attributes: 比较品牌的属性
    :return: 唯一性得分字典
    """
    uniqueness = {}

    all_attrs = set(brand_attributes.keys()) | set(compare_brand_attributes.keys())

    for attr in all_attrs:
        # 获取两个品牌的NWCD值
        nwcd_focal = brand_attributes.get(attr, {}).get('nwcd', 0)
        nwcd_compare = compare_brand_attributes.get(attr, {}).get('nwcd', 0)

        # 避免除以零
        max_nwcd = max(nwcd_focal, nwcd_compare)
        if max_nwcd == 0:
            uniqueness[attr] = 0
            continue

        # 按照论文公式计算唯一性
        uniqueness[attr] = (nwcd_focal - nwcd_compare) / max_nwcd

    return uniqueness


def visualize_brand_uniqueness(uniqueness_data, brand1, brand2, save_path=None):
    """
    可视化品牌属性唯一性，类似论文中的表格
    :param uniqueness_data: 唯一性数据字典
    :param brand1: 第一个品牌名称
    :param brand2: 第二个品牌名称
    :param save_path: 保存路径
    """
    print(f"正在生成 {brand1} vs {brand2} 的唯一性可视化...")

    # 获取可用的中文字体
    chinese_font = setup_fonts()

    # 选择前10个最具唯一性的属性（正负向各10个）
    sorted_uniqueness = sorted(uniqueness_data.items(), key=lambda x: x[1], reverse=True)
    positive_unique = sorted_uniqueness[:10]  # brand1相对于brand2最具唯一性的属性
    negative_unique = sorted(sorted_uniqueness, key=lambda x: x[1])[:10]  # brand2相对于brand1最具唯一性的属性

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=100)
    fig.patch.set_facecolor('#f8f9fa')

    # 设置标题和样式
    fig.suptitle(f'{brand1} vs {brand2} 品牌属性唯一性对比', fontsize=18, fontfamily=chinese_font)

    # 处理brand1唯一属性
    brand1_attrs = [attr for attr, _ in positive_unique]
    brand1_values = [val for _, val in positive_unique]
    brand1_colors = ['#3498db' if val > 0.5 else '#5dade2' if val > 0.3 else '#85c1e9' for val in brand1_values]

    # 处理brand2唯一属性
    brand2_attrs = [attr for attr, _ in negative_unique]
    brand2_values = [abs(val) for _, val in negative_unique]  # 取绝对值来显示
    brand2_colors = ['#e74c3c' if val > 0.5 else '#ec7063' if val > 0.3 else '#f1948a' for val in brand2_values]

    # 绘制brand1唯一属性
    y_pos1 = np.arange(len(brand1_attrs))
    ax1.barh(y_pos1, brand1_values, color=brand1_colors, edgecolor='white', height=0.6)
    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels(brand1_attrs, fontfamily=chinese_font, fontsize=10)
    ax1.set_title(f'{brand1} 独特属性', fontsize=14, fontfamily=chinese_font)
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', linestyle='--', alpha=0.3)

    # 绘制brand2唯一属性
    y_pos2 = np.arange(len(brand2_attrs))
    ax2.barh(y_pos2, brand2_values, color=brand2_colors, edgecolor='white', height=0.6)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(brand2_attrs, fontfamily=chinese_font, fontsize=10)
    ax2.set_title(f'{brand2} 独特属性', fontsize=14, fontfamily=chinese_font)
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', linestyle='--', alpha=0.3)

    # 添加说明文字
    fig.text(0.5, 0.01, '唯一性得分 (0-1，越大表示属性越独特)', ha='center', fontsize=12, fontfamily=chinese_font)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"  唯一性图表已保存到: {save_path}")

    plt.show()


# 导出品牌属性信息到Excel
def export_brand_attributes_to_excel(brand_attributes, output_file='brand_attributes_analysis.xlsx',
                                     max_attrs_per_brand=50):
    """
    将所有品牌的属性信息导出到Excel文件
    :param brand_attributes: 品牌属性字典
    :param output_file: 输出Excel文件路径
    :param max_attrs_per_brand: 每个品牌最多展示的属性数量
    :return: 导出的DataFrame
    """
    print(f"正在导出所有品牌属性信息到: {output_file} (每个品牌最多{max_attrs_per_brand}个属性)")

    # 创建一个空的DataFrame
    df = pd.DataFrame()

    # 目标词性比例
    target_noun_ratio = 0.6
    target_adj_ratio = 0.4

    # 按品牌逐个处理
    for brand_name in sorted(brand_attributes.keys()):
        attributes = brand_attributes[brand_name]

        # 分离名词和形容词属性
        noun_attrs = {attr: info for attr, info in attributes.items() if info.get('pos') == 'NN'}
        adj_attrs = {attr: info for attr, info in attributes.items() if info.get('pos') == 'JJ'}

        # 按中心度排序
        sorted_nouns = sorted(noun_attrs.items(), key=lambda x: x[1]['nwcd'], reverse=True)
        sorted_adjs = sorted(adj_attrs.items(), key=lambda x: x[1]['nwcd'], reverse=True)

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
                '词性': '名词' if info.get('pos') == 'NN' else '形容词' if info.get('pos') == 'JJ' else '未知',
                '总计数': info['count'],
                '情感类型': info['sentiment'],
                '情感得分': info['sentiment_score'],
                '正面计数': info['positive_count'],
                '负面计数': info['negative_count'],
                '中性计数': info['neutral_count'],
                '情感强度': info['strength'],
                '中心度': info.get('nwcd', 0),
                '文档频率': info.get('doc_freq', 0),
                'TF-IDF得分': info.get('tf_idf', 0)
            }
            brand_data.append(row_data)

        # 将这个品牌的数据追加到主DataFrame
        brand_df = pd.DataFrame(brand_data)
        df = pd.concat([df, brand_df], ignore_index=True)

    # 保存到Excel
    try:
        # 创建一个Excel writer对象
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

        # 将数据写入Excel
        df.to_excel(writer, index=False, sheet_name='品牌属性分析')

        # 获取xlsxwriter对象
        workbook = writer.book
        worksheet = writer.sheets['品牌属性分析']

        # 添加格式
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
        print(f"品牌属性信息已导出到: {output_file}")
    except Exception as e:
        print(f"保存Excel出错: {e}")
        # 如果导出失败，尝试普通导出
        df.to_excel(output_file, index=False)
        print(f"品牌属性信息已使用标准格式导出到: {output_file}")

    return df


# 主函数 - 实现论文中的完整流程
def main():
    """
    主函数 - 实现基于人类联想记忆模型的品牌形象分析
    """
    # 1. 加载预处理后的数据，并按品牌分组
    processed_df = load_processed_data_by_brand('processed_reviews.xlsx', brand_mapping)
    print(f"加载了 {len(processed_df)} 条评论记录")

    # 2. 提取品牌属性
    brand_attributes = extract_brand_attributes(processed_df, min_frequency=3, min_word_length=2)

    # 3. 计算属性中心度
    brand_attributes = calculate_brand_centrality(brand_attributes, processed_df)

    # 4. 识别属性间关联
    connections = identify_brand_attribute_connections(processed_df, brand_attributes)

    # 5. 为每个品牌构建并可视化网络
    for brand_name in brand_attributes:
        if len(brand_attributes[brand_name]) > 5:  # 只处理有足够属性的品牌
            # 使用人类联想记忆模型构建品牌关联网络
            G = build_brand_network(brand_name, brand_attributes, connections,
                                    min_edge_weight=2, top_attributes=80)

            title = f"{brand_name} 品牌评论属性关联网络"
            save_path = f"{brand_name}_brand_network_ham.png"  # HAM表示人类联想记忆模型
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

        # 6. 比较两个品牌的唯一性 (如果有两个品牌)
        if '小米' in brand_attributes and '特斯拉' in brand_attributes:
            print("\n比较比亚迪和特斯拉的品牌属性唯一性...")

            # 计算唯一性
            byd_uniqueness = calculate_uniqueness(
                brand_attributes['小米'],
                brand_attributes['特斯拉']
            )

            # 输出最具唯一性的属性
            sorted_uniqueness = sorted(byd_uniqueness.items(), key=lambda x: x[1], reverse=True)

            print("\n小米相对于特斯拉最具唯一性的属性:")
            for attr, score in sorted_uniqueness[:10]:
                if score > 0.3:  # 只显示唯一性大于0.3的属性
                    print(f"  - {attr}: 唯一性得分: {score:.4f}")

            # 反向唯一性（特斯拉相对于比亚迪的唯一性）
            print("\n特斯拉相对于小米最具唯一性的属性:")
            for attr, score in sorted(sorted_uniqueness, key=lambda x: x[1])[:10]:
                if score < -0.3:  # 负值表示特斯拉更具唯一性
                    print(f"  - {attr}: 唯一性得分: {score:.4f}")

            # 可视化唯一性对比 (新增功能)
            visualize_brand_uniqueness(byd_uniqueness, '小米', '特斯拉', '小米_特斯拉_uniqueness.png')

    # 7. 导出所有品牌属性到Excel
    export_df = export_brand_attributes_to_excel(brand_attributes)
    print("\n分析完成!")


# 按照论文方法，分析消费者随时间变化的品牌认知
def analyze_brand_image_over_time(processed_df, brand_name, time_column,
                                  period1_start, period1_end,
                                  period2_start, period2_end):
    """
    分析品牌形象随时间的变化
    :param processed_df: 处理后的数据
    :param brand_name: 品牌名称
    :param time_column: 时间列
    :param period1_start: 第一个时间段的开始
    :param period1_end: 第一个时间段的结束
    :param period2_start: 第二个时间段的开始
    :param period2_end: 第二个时间段的结束
    """
    print(f"分析 {brand_name} 品牌形象随时间变化 ({period1_start}-{period1_end} vs {period2_start}-{period2_end})...")

    # 提取该品牌的数据
    brand_df = processed_df[processed_df['brand'] == brand_name]

    # 过滤两个时间段的数据
    period1_df = brand_df[(brand_df[time_column] >= period1_start) &
                          (brand_df[time_column] <= period1_end)]
    period2_df = brand_df[(brand_df[time_column] >= period2_start) &
                          (brand_df[time_column] <= period2_end)]

    print(f"  时间段1: {period1_start}-{period1_end}, 评论数: {len(period1_df)}")
    print(f"  时间段2: {period2_start}-{period2_end}, 评论数: {len(period2_df)}")

    # 提取两个时间段的属性
    min_frequency = 3
    min_word_length = 2

    # 创建临时数据框
    temp_df = pd.DataFrame({
        'brand': brand_name,
        'model': brand_name,  # 简化模型处理
        'review_text': [[]]
    })

    # 为两个时间段单独提取属性
    period1_attributes = extract_brand_attributes(
        pd.concat([temp_df, period1_df]),
        min_frequency, min_word_length
    )[brand_name]

    period2_attributes = extract_brand_attributes(
        pd.concat([temp_df, period2_df]),
        min_frequency, min_word_length
    )[brand_name]

    # 计算属性中心度
    period1_attributes = calculate_brand_centrality({brand_name: period1_attributes}, period1_df)[brand_name]
    period2_attributes = calculate_brand_centrality({brand_name: period2_attributes}, period2_df)[brand_name]

    # 比较两个时间段的属性变化
    print("\n两个时间段之间的属性变化:")

    # 获取所有属性
    all_attrs = set(period1_attributes.keys()) | set(period2_attributes.keys())

    # 计算变化
    changes = []
    for attr in all_attrs:
        nwcd1 = period1_attributes.get(attr, {}).get('nwcd', 0)
        nwcd2 = period2_attributes.get(attr, {}).get('nwcd', 0)

        if nwcd1 == 0 and nwcd2 > 0:
            # 新出现的属性
            changes.append((attr, "新出现", 0, nwcd2))
        elif nwcd1 > 0 and nwcd2 == 0:
            # 消失的属性
            changes.append((attr, "消失", nwcd1, 0))
        else:
            # 变化的属性
            change = nwcd2 - nwcd1
            percent_change = (change / nwcd1 * 100) if nwcd1 > 0 else 0

            if abs(percent_change) >= 20:  # 仅显示变化显著的属性
                direction = "增加" if change > 0 else "减少"
                changes.append((attr, direction, nwcd1, nwcd2))

    # 按照变化幅度排序
    changes.sort(key=lambda x: abs(x[3] - x[2]), reverse=True)

    # 输出结果
    for attr, direction, nwcd1, nwcd2 in changes[:20]:  # 显示前20个变化最显著的属性
        if direction == "新出现":
            sentiment = period2_attributes.get(attr, {}).get('sentiment', '未知')
            print(f"  - {attr} (新出现): 中心度: {nwcd2:.4f}, 情感: {sentiment}")
        elif direction == "消失":
            sentiment = period1_attributes.get(attr, {}).get('sentiment', '未知')
            print(f"  - {attr} (消失): 中心度: {nwcd1:.4f}, 情感: {sentiment}")
        else:
            change = nwcd2 - nwcd1
            percent_change = (change / nwcd1 * 100) if nwcd1 > 0 else 0
            sentiment1 = period1_attributes.get(attr, {}).get('sentiment', '未知')
            sentiment2 = period2_attributes.get(attr, {}).get('sentiment', '未知')

            sentiment_change = ""
            if sentiment1 != sentiment2:
                sentiment_change = f", 情感变化: {sentiment1} -> {sentiment2}"

            print(
                f"  - {attr} ({direction}): 中心度: {nwcd1:.4f} -> {nwcd2:.4f}, 变化: {percent_change:.1f}%{sentiment_change}")

    # 返回属性变化
    return {
        'period1': period1_attributes,
        'period2': period2_attributes,
        'changes': changes
    }


# 分析不同消费者群体对品牌的认知差异
def analyze_brand_image_across_consumers(processed_df, brand_name, segment_column, segment_values):
    """
    分析不同消费者群体对品牌的认知差异
    :param processed_df: 处理后的数据
    :param brand_name: 品牌名称
    :param segment_column: 用于分段的列（例如 'recommendation'）
    :param segment_values: 分段值字典 {'segment_name': segment_value}
    """
    print(f"分析不同消费者群体对 {brand_name} 的认知差异...")

    # 提取该品牌的数据
    brand_df = processed_df[processed_df['brand'] == brand_name]

    # 分析每个消费者群体
    segment_attributes = {}

    for segment_name, segment_value in segment_values.items():
        # 过滤该细分市场的数据
        segment_df = brand_df[brand_df[segment_column] == segment_value]
        print(f"  消费者群体 '{segment_name}': {len(segment_df)} 条评论")

        if len(segment_df) == 0:
            print(f"  警告: 消费者群体 '{segment_name}' 没有数据")
            continue

        # 创建临时数据框
        temp_df = pd.DataFrame({
            'brand': brand_name,
            'model': brand_name,  # 简化模型处理
            'review_text': [[]]
        })

        # 提取该消费者群体的属性
        segment_attrs = extract_brand_attributes(
            pd.concat([temp_df, segment_df]),
            min_frequency=2,
            min_word_length=2
        )[brand_name]

        # 计算属性中心度
        segment_attrs = calculate_brand_centrality({brand_name: segment_attrs}, segment_df)[brand_name]

        # 存储该消费者群体的属性
        segment_attributes[segment_name] = segment_attrs

        # 输出该消费者群体的主要属性
        print(f"\n消费者群体 '{segment_name}' 的主要属性:")

        print("正面属性:")
        sorted_positive = sorted([i for i in segment_attrs.items()
                                  if i[1]['sentiment'] == 'positive'],
                                 key=lambda x: x[1]['nwcd'], reverse=True)

        for attr, info in sorted_positive[:5]:  # 显示前5个正面属性
            print(f"  - {attr}: 中心度: {info['nwcd']:.4f}, 计数: {info['count']}")

        print("负面属性:")
        sorted_negative = sorted([i for i in segment_attrs.items()
                                  if i[1]['sentiment'] == 'negative'],
                                 key=lambda x: x[1]['nwcd'], reverse=True)

        for attr, info in sorted_negative[:5]:  # 显示前5个负面属性
            print(f"  - {attr}: 中心度: {info['nwcd']:.4f}, 计数: {info['count']}")

    # 如果有多个消费者群体，比较它们之间的差异
    if len(segment_attributes) > 1:
        segment_names = list(segment_attributes.keys())

        for i in range(len(segment_names)):
            for j in range(i + 1, len(segment_names)):
                segment1 = segment_names[i]
                segment2 = segment_names[j]

                print(f"\n比较消费者群体 '{segment1}' 和 '{segment2}' 之间的差异:")

                # 计算唯一性
                uniqueness = calculate_uniqueness(
                    segment_attributes[segment1],
                    segment_attributes[segment2]
                )

                # 输出最具唯一性的属性
                sorted_uniqueness = sorted(uniqueness.items(), key=lambda x: x[1], reverse=True)

                print(f"'{segment1}' 相对于 '{segment2}' 最具唯一性的属性:")
                for attr, score in sorted_uniqueness[:5]:
                    if score > 0.3:  # 只显示唯一性大于0.3的属性
                        print(f"  - {attr}: 唯一性得分: {score:.4f}")

                print(f"'{segment2}' 相对于 '{segment1}' 最具唯一性的属性:")
                for attr, score in sorted(sorted_uniqueness, key=lambda x: x[1])[:5]:
                    if score < -0.3:  # 负值表示segment2更具唯一性
                        print(f"  - {attr}: 唯一性得分: {score:.4f}")

    return segment_attributes


# 分析特定产品对品牌形象的贡献
def analyze_product_contribution(processed_df, brand_name, attribute_to_analyze):
    """
    分析特定产品对品牌形象的贡献
    :param processed_df: 处理后的数据
    :param brand_name: 品牌名称
    :param attribute_to_analyze: 待分析的属性
    """
    print(f"分析 {brand_name} 各产品对属性 '{attribute_to_analyze}' 的贡献...")

    # 提取该品牌的数据
    brand_df = processed_df[processed_df['brand'] == brand_name]

    # 获取所有提及该属性的评论
    attribute_reviews = brand_df[brand_df['words'].apply(lambda x: attribute_to_analyze in x)]

    # 按产品统计该属性的出现次数
    product_counts = attribute_reviews['model'].value_counts()

    # 统计每个产品对该属性的正面/负面评价
    product_sentiment = {}

    for product in product_counts.index:
        product_reviews = attribute_reviews[attribute_reviews['model'] == product]

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for _, row in product_reviews.iterrows():
            # 判断该评论对属性的情感倾向
            # 如果包含review_section
            if 'review_section' in row:
                if row['review_section'] == 'pros':
                    positive_count += 1
                else:
                    negative_count += 1
            else:
                # 使用情感概率
                sentiment = 'positive' if row['positive_prob'] > row['negative_prob'] else 'negative'
                if sentiment == 'positive':
                    positive_count += 1
                else:
                    negative_count += 1

        # 计算正面情感占比
        total = positive_count + negative_count + neutral_count
        if total > 0:
            positive_ratio = positive_count / total
        else:
            positive_ratio = 0

        product_sentiment[product] = {
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count,
            'total': total,
            'positive_ratio': positive_ratio
        }

    # 输出结果
    print(f"\n属性 '{attribute_to_analyze}' 在各产品中的表现:")

    # 按照正面情感占比排序
    sorted_products = sorted(product_sentiment.items(), key=lambda x: x[1]['positive_ratio'], reverse=True)

    for product, info in sorted_products:
        if info['total'] >= 5:  # 只显示评论数达到一定阈值的产品
            print(
                f"  - {product}: 总计: {info['total']}条, 正面: {info['positive']}条 ({info['positive_ratio']:.1%}), 负面: {info['negative']}条 ({1 - info['positive_ratio']:.1%})")

    return product_sentiment


# 示例
def full_analysis_example():
    """
    执行完整的分析示例
    """
    print("=== 基于人类联想记忆模型的品牌形象分析示例 ===\n")

    # 1. 加载数据
    processed_df = load_processed_data_by_brand('processed_reviews.xlsx', brand_mapping)

    # 2. 主要品牌分析
    main()

    # 3. 时间序列分析 (示例)
    if 'date' in processed_df.columns:
        analyze_brand_image_over_time(
            processed_df,
            '比亚迪',
            'date',
            '2020-01-01', '2021-12-31',  # 第一个时间段
            '2022-01-01', '2023-12-31'  # 第二个时间段
        )

    # 4. 消费者群体分析 (示例)
    if 'recommendation' in processed_df.columns:
        analyze_brand_image_across_consumers(
            processed_df,
            '比亚迪',
            'recommendation',
            {'推荐者': True, '不推荐者': False}
        )

    # 5. 产品贡献分析 (示例)
    analyze_product_contribution(processed_df, '比亚迪', '操控')

    print("\n=== 分析完成 ===")


# 运行主函数
if __name__ == "__main__":
    main()
    # 如需运行完整示例，取消下一行注释
    # full_analysis_example()
