import multiprocessing
import logging
# 创建一个logger
logger = logging.getLogger('profile')
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于将日志写入文件
fh = logging.FileHandler('/data/home/josephyou/WXG_WORK/sglang/python/sglang/srt/profile/profile.log')
fh.setLevel(logging.DEBUG)


# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)


class ProcessScheduler:
    def __init__(self) -> None:
        # 创建一个Manager对象
        self.manager = multiprocessing.Manager()

        # 创建一个ProfileData实例
        self.shared_object = self.manager.Namespace()

        self.shared_object.mem_data = self.manager.list()
        self.shared_object.batch_data = self.manager.list()
        self.shared_object.type = self.manager.list()
        self.shared_object.max_size = 0.0

    
    def create_process(self, target, args):
        process = multiprocessing.Process(target=target, args=args)
        return process
    
    def start_process(self, process):
        process.start()

    def join_process(self, process):
        process.join()

    def get_shared_object(self):
        return self.shared_object
    

    def print_profile_data(self):
      

        # 记录一些日志

        logger.info(f"id\t\tmem\t\tbatch\t\ttype")
        length = min(len(self.shared_object.mem_data), len(self.shared_object.batch_data), len(self.shared_object.type))
        for i in range(0, length):
            logger.info(f"{i}\t\t{self.shared_object.mem_data[i]}\t\t{self.shared_object.batch_data[i]}\t\t{self.shared_object.type[i]}")
    

    def plot_profile_data(self):
        import matplotlib.pyplot as plt

        gpu_memory_usage_percentage = [(float(self.shared_object.max_size - usage) / self.shared_object.max_size) * 100 for usage in self.shared_object.mem_data]
        compute_resource_usage = [(float(usage) / 512) * 100 for usage in self.shared_object.batch_data]

        plt.figure(figsize=(10, 6))

        # length = min(len(shared_object.mem_data), len(shared_object.batch_data), len(shared_object.type))
        plt.plot(gpu_memory_usage_percentage, label='GPU Memory Usage (%)')
        plt.plot(compute_resource_usage, label='Compute Resource Usage (%)')
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Rate(%)")
        plt.savefig('/data/home/josephyou/WXG_WORK/sglang/python/sglang/srt/profile/profile.png')
        
        print("memroy use saved...")