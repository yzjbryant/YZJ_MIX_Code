import os
path='F:\奈园代码\p6.3——key_29个指标'
Folder=["高职创新发展行动计划骨干专业",	"高职创新发展行动计划生产性实训基地","高职创新发展行动计划优质专科高等职业院校",	"高职创新发展行动计划双师基地",
        "高职创新发展行动计划虚拟仿真实训中心","高职创新发展行动计划协同创新中心","高职创新发展行动计划技能大师工作室",	"李四光优秀学生奖","人民网奖学金",
        "吴瑞奖学金",	"全国大学生“小平科技创新团队”","中国大学生年度人物","中国大学生自强之星标兵","中国大学生自强之星","中国青少年科技创新奖","全国大学生年度创新人物",
        "全国高校云计算应用创新大赛","全国高校互联网金融应用创新大赛",	"中国大学生医学技术技能大赛","全国职业院校技能大赛职业院校教学能力比赛获奖名单",
        "全国职业院校信息化教学大赛获奖名单","全国大学生智能汽车竞赛",	"全国高等医学院校临床基本技能竞赛",	"全国大学生结构设计竞赛","全国大学生机械创新设计大赛",
        "全国研究生数学建模竞赛奖","全国大学生桥牌锦标赛","全国大学生物流设计大赛",	"全国大学生广告艺术大赛"]
Folder2=["处理数据","原始数据"]
for i in range(len(Folder)):
    os.chdir(path)
    # os.makedirs(str(Folder[i]))
    mkpath=Folder[i]+'/'
    # wenben=open(mkpath+str(Folder[i])+".xlsx","w")
    os.chdir(mkpath)
    os.makedirs("原始数据")
    os.makedirs("处理结果")