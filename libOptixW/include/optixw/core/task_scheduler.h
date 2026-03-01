// UTF-8 BOM - 确保 MSVC 正确识别中文注释
#pragma once

#include <taskflow/taskflow.hpp>
#include <memory>
#include <vector>

namespace optixw {

// TaskScheduler - CPU 层任务调度器
// 封装 Taskflow 执行器，用于并行化 CPU 端任务
// 职责：
//   - 管理 Taskflow 执行器和线程池
//   - 提供任务流创建和执行接口
//   - 支持同步和异步执行
class TaskScheduler {
public:
    // 构造函数
    // numThreads: 线程池大小，0 表示自动检测（使用硬件并发数）
    explicit TaskScheduler(size_t numThreads = 0);
    
    ~TaskScheduler();

    // 创建新的任务流
    // 返回的任务流由 TaskScheduler 管理生命周期
    tf::Taskflow& createTaskflow();

    // 同步执行任务流（阻塞直到完成）
    void run(tf::Taskflow& taskflow);

    // 异步执行任务流
    // 返回 Future 对象，可用于等待完成
    tf::Future<void> runAsync(tf::Taskflow& taskflow);

    // 等待所有异步任务完成
    void waitForAll();

    // 获取线程池大小
    size_t getNumThreads() const;

    // 获取 Taskflow 执行器（供高级用户使用）
    tf::Executor& getExecutor() { return m_executor; }
    const tf::Executor& getExecutor() const { return m_executor; }

private:
    // Taskflow 执行器（管理线程池）
    tf::Executor m_executor;

    // 任务流容器（管理生命周期）
    std::vector<std::unique_ptr<tf::Taskflow>> m_taskflows;

    // 禁止拷贝和赋值
    TaskScheduler(const TaskScheduler&) = delete;
    TaskScheduler& operator=(const TaskScheduler&) = delete;
};

} // namespace optixw
