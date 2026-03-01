// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#include <optixw/core/task_scheduler.h>
#include <thread>

namespace optixw {

TaskScheduler::TaskScheduler(size_t numThreads)
    : m_executor(numThreads == 0 ? std::thread::hardware_concurrency() : numThreads)
{
    // 构造函数：初始化 Taskflow 执行器
    // 如果 numThreads 为 0，使用硬件并发数（通常是 CPU 核心数）
}

TaskScheduler::~TaskScheduler() {
    // 析构函数：等待所有任务完成
    waitForAll();
}

tf::Taskflow& TaskScheduler::createTaskflow() {
    // 创建新的任务流并返回引用
    m_taskflows.push_back(std::make_unique<tf::Taskflow>());
    return *m_taskflows.back();
}

void TaskScheduler::run(tf::Taskflow& taskflow) {
    // 同步执行任务流（阻塞直到完成）
    m_executor.run(taskflow).wait();
}

tf::Future<void> TaskScheduler::runAsync(tf::Taskflow& taskflow) {
    // 异步执行任务流，返回 Future 对象
    return m_executor.run(taskflow);
}

void TaskScheduler::waitForAll() {
    // 等待所有异步任务完成
    m_executor.wait_for_all();
}

size_t TaskScheduler::getNumThreads() const {
    // 获取线程池大小
    return m_executor.num_workers();
}

} // namespace optixw
