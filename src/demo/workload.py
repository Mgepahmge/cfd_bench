from ..Experiment_Script_w1 import main as run_w1
from ..Experiment_Script_w2 import main as run_w2
from ..Experiment_Script_w3 import main as run_w3
from ..Experiment_Script_w4 import main as run_w4
from ..Experiment_Script_w5 import main as run_w5
from ..Experiment_Script_w6 import main as run_w6
from ..Experiment_Script_w7 import main as run_w7
from ..Experiment_Script_w8 import main as run_w8

def main():
    
    # 1. 在这里统一修改要测试的船型列表
    target_ships = ["Kvlcc2_351k"] 
    # target_ships = ["JBC_615k", "Kvlcc2_351k", "Suboff_3258k"]

    # 2. 依次执行
    run_w1(target_ships)
    run_w2(target_ships)
    run_w3(target_ships)
    run_w4(target_ships)
    run_w5(target_ships)
    run_w6(target_ships)
    run_w7(target_ships)
    run_w8(target_ships)

    # 3. 或者直接执行全部（默认测试所有船型）
    # run_w1()
    # run_w2()
    # run_w3()
    # run_w4()
    # run_w5()
    # run_w6()
    # run_w7()
    # run_w8()    

    return

if __name__ == "__main__":
    main()