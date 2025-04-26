"""Simple Assistant Agent."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import re
import json
import decimal

from dbgpt.rag.retriever.rerank import RetrieverNameRanker
from dbgpt._private.pydantic import BaseModel, Field, model_to_json

from .. import AgentMessage
from ..core.action.base import Action, ActionOutput
from ..core.action.blank_action import BlankAction
from ..core.base_agent import ConversableAgent
from ..core.profile import DynConfig, ProfileConfig
from ..resource.base import AgentResource, ResourceType
from ..resource.database import DBResource
from ...vis.base import Vis
from ...vis.tags.vis_chart import VisChart

logger = logging.getLogger(__name__)


class SqlInput(BaseModel):
    """SQL input model."""
    display_type: str = Field(..., description="图表渲染方法")
    sql: str = Field(..., description="可执行的SQL")
    thought: str = Field(..., description="分析思路")


class MultiDimensionSqlInput(BaseModel):
    """多维度SQL分析模型"""
    dimensions: List[Dict[str, Any]] = Field(
        ...,
        description="多个维度的分析，每个维度包含display_type、sql和thought"
    )


class SimpleChartAction(Action[SqlInput]):
    """Simple chart action class for SimpleAssistantAgent."""

    def __init__(self, **kwargs):
        """Chart action init."""
        super().__init__(**kwargs)
        self._render_protocol = VisChart()

    @property
    def resource_need(self) -> Optional[ResourceType]:
        """Return the resource type needed for the action."""
        return ResourceType.DB

    @property
    def render_protocol(self) -> Optional[Vis]:
        """Return the render protocol."""
        return self._render_protocol

    @property
    def out_model_type(self):
        """Return the output model type."""
        return SqlInput
        
    def render_prompt(self) -> Optional[str]:
        """Return the render prompt with chart types."""
        antv_charts = [
            {"response_line_chart": "用于显示比较趋势分析数据"},
            {"response_pie_chart": "适用于比例和分布统计等场景"},
            {"response_table": "适用于展示多列数据或非数值列"},
            {"response_scatter_plot": "适用于探索变量间关系、检测异常值等"},
            {"response_bubble_chart": "适用于表示多变量关系、突出异常情况等"},
            {"response_donut_chart": "适用于层次结构表示、类别比例显示等"},
            {"response_area_chart": "适用于时间序列数据可视化、多组数据比较等"},
            {"response_heatmap": "适用于时间序列数据可视化、大规模数据集、分类数据分布等"},
            {"response_column_chart": "适用于分类数据比较、时间序列分析等"}
        ]
        display_type = "\n".join(
            f"{key}:{value}" for dict_item in antv_charts for key, value in dict_item.items()
        )
        return display_type
        
    def fix_sql_dialect(self, sql, dialect):
        """针对特定数据库方言修复SQL"""
        if not dialect or dialect.lower() != "mysql":
            return sql
        
        logger.info(f"开始SQL方言修正，数据库类型: {dialect}")
        
        try:
            # 处理GROUP BY中的别名问题
            # 在MySQL中不能使用列别名，需要使用完整表达式
            group_pattern = re.compile(r'GROUP\s+BY\s+(.*?)(?:ORDER BY|LIMIT|HAVING|$)', re.IGNORECASE | re.DOTALL)
            group_match = group_pattern.search(sql)
            
            if group_match:
                group_cols = group_match.group(1).strip()
                # 查找所有SELECT项中的别名定义
                select_pattern = re.compile(r'SELECT\s+(.+?)\s+FROM', re.IGNORECASE | re.DOTALL)
                select_match = select_pattern.search(sql)
                
                if select_match:
                    select_items = select_match.group(1).split(',')
                    aliases = {}
                    
                    # 提取所有别名定义
                    for item in select_items:
                        as_match = re.search(r'(.+?)\s+AS\s+[\'\"]?([^\'\",]+)[\'\"]?', item.strip(), re.IGNORECASE)
                        if as_match:
                            expression = as_match.group(1).strip()
                            alias = as_match.group(2).strip()
                            aliases[alias] = expression
                    
                    # 替换GROUP BY中的别名
                    group_cols_parts = [col.strip() for col in group_cols.split(',')]
                    for i, part in enumerate(group_cols_parts):
                        if part in aliases:
                            group_cols_parts[i] = aliases[part]
                    
                    # 重构SQL
                    new_group_clause = "GROUP BY " + ", ".join(group_cols_parts)
                    sql = group_pattern.sub(new_group_clause + (' ' if group_match.group(0).endswith(' ') else ''), sql)
            
            # 同样处理HAVING子句中的别名
            having_pattern = re.compile(r'HAVING\s+(.*?)(?:ORDER BY|LIMIT|$)', re.IGNORECASE | re.DOTALL)
            having_match = having_pattern.search(sql)
            
            if having_match:
                having_clause = having_match.group(1).strip()
                select_pattern = re.compile(r'SELECT\s+(.+?)\s+FROM', re.IGNORECASE | re.DOTALL)
                select_match = select_pattern.search(sql)
                
                if select_match:
                    select_items = select_match.group(1).split(',')
                    aliases = {}
                    
                    # 提取所有别名定义
                    for item in select_items:
                        as_match = re.search(r'(.+?)\s+AS\s+[\'\"]?([^\'\",]+)[\'\"]?', item.strip(), re.IGNORECASE)
                        if as_match:
                            expression = as_match.group(1).strip()
                            alias = as_match.group(2).strip()
                            aliases[alias] = expression
                    
                    # 替换HAVING中的别名
                    for alias, expr in aliases.items():
                        having_clause = re.sub(r'\b' + re.escape(alias) + r'\b', f'({expr})', having_clause)
                    
                    # 重构SQL
                    new_having_clause = "HAVING " + having_clause
                    sql = having_pattern.sub(new_having_clause + (' ' if having_match.group(0).endswith(' ') else ''), sql)
            
            logger.info(f"修正后完整SQL: {sql}")
            return sql
            
        except Exception as e:
            logger.error(f"SQL方言修正过程中出现错误: {str(e)}", exc_info=True)
            return sql

    async def run(
        self,
        ai_message: str,
        resource: Optional[AgentResource] = None,
        rely_action_out: Optional[ActionOutput] = None,
        need_vis_render: bool = True,
        **kwargs,
    ) -> ActionOutput:
        """处理简单助手的图表请求。"""
        try:
            db_resources: List[DBResource] = DBResource.from_resource(self.resource)
            if not db_resources:
                return ActionOutput(is_exe_success=True, content=ai_message, view=ai_message)
            
            db = db_resources[0]
            
            # 首先尝试解析JSON格式
            json_pattern = re.compile(r'\{\s*"display_type":\s*"([^"]+)",\s*"sql":\s*"([^"]+)",\s*"thought":\s*"([^"]+)"\s*\}', re.DOTALL)
            json_matches = json_pattern.findall(ai_message)
            
            # 添加SQL语法检查和修复
            def check_and_fix_sql(sql):
                # 检查括号匹配
                open_brackets = sql.count('(')
                close_brackets = sql.count(')')
                if open_brackets > close_brackets:
                    # 尝试修复缺少的右括号
                    fixed_sql = sql.replace(' FROM ', ')' * (open_brackets - close_brackets) + ' FROM ', 1)
                    logger.info(f"SQL语法修复: 添加{open_brackets - close_brackets}个右括号")
                    return fixed_sql
                return sql
            
            # 如果找到JSON格式的匹配
            if json_matches:
                report_sections = []
                success_dimensions = []  # 跟踪成功执行的维度
                
                # 1. 创建报告标题
                report_title = "# 营销活动用户群体分析报告"
                
                # 2. 处理每个维度作为一个章节
                for i, (display_type, sql, thought) in enumerate(json_matches):
                    try:
                        # SQL执行和图表生成
                        sql = sql.replace("\\", "")
                        fixed_sql = check_and_fix_sql(sql)
                        fixed_sql = self.fix_sql_dialect(fixed_sql, db.dialect)
                        
                        # 添加更详细的日志
                        logger.info(f"正在执行SQL: {fixed_sql}")
                        data_df = await db.query_to_df(fixed_sql)
                        
                        # 创建章节标题和内容
                        section_number = i + 1
                        section_title = f"## {section_number}. {get_dimension_name(i)}"
                        section_content = f"### {section_number}.1 分析结论\n{thought}\n\n"
                        
                        # 记录章节内容
                        report_sections.append(f"{section_title}\n{section_content}")
                        
                        # 记录成功维度
                        success_dimensions.append((i, display_type, fixed_sql, thought, data_df))
                        logger.info(f"维度{i}：视图生成成功，数据长度: {len(data_df) if data_df is not None else 0}")
                        
                    except Exception as e:
                        # 报错处理 - 只记录日志，不添加到chart_views
                        error_msg = f"图表生成失败: {str(e)}"
                        logger.error(error_msg)
                        
                        # 仍然包含分析结论，但标记图表失败
                        section_number = i + 1
                        section_title = f"## {section_number}. {get_dimension_name(i)}"
                        section_content = f"### {section_number}.1 分析结论\n{thought}\n\n### {section_number}.2 注意：图表生成失败\n"
                        report_sections.append(f"{section_title}\n{section_content}")
                
                # 3. 添加后续建议部分
                conclusion_section = "\n## 后续建议\n根据以上分析,建议关注高价值客户群体,优化营销策略。"
                report_sections.append(conclusion_section)
                
                # 组合完整报告
                full_report = f"{report_title}\n\n" + "\n\n".join(report_sections)
                
                # 创建符合前端期望格式的结构
                structured_view = {
                    "vis_type": "vis-dashboard",
                    "vis_content": {
                        "title": "营销活动用户群体分析报告",
                        "display_strategy": "report", 
                        "chart_count": len(success_dimensions),
                        "data": []
                    }
                }
                
                # 添加图表数据
                for idx, (i, display_type, sql, thought, data_df) in enumerate(success_dimensions):
                    # 转换DataFrame为记录列表
                    records = []
                    if data_df is not None and not data_df.empty:
                        for _, row in data_df.iterrows():
                            record = {}
                            for column in data_df.columns:
                                # 确保值是简单类型
                                value = row[column]
                                if isinstance(value, (int, float, str, bool)) or value is None:
                                    record[column] = value
                                elif isinstance(value, decimal.Decimal):
                                    record[column] = float(value)
                                else:
                                    record[column] = str(value)  # 转换复杂对象为字符串
                            records.append(record)
                    
                    chart_item = {
                        "type": display_type,  # 使用type而非display_type
                        "title": f"{idx+1}. {get_dimension_name(i)}",
                        "describe": thought,
                        "sql": sql,
                        "data": records
                    }
                    
                    structured_view["vis_content"]["data"].append(chart_item)
                    logger.info(f"成功构建视图数据, 包含{len(records)}条记录")
                
                # 确保添加至少一个默认数据，避免完全空白
                if not structured_view["vis_content"]["data"]:
                    structured_view["vis_content"]["data"].append({
                        "type": "response_table",
                        "title": "数据处理结果",
                        "describe": "所有SQL查询执行失败，请检查SQL语法或数据库连接。",
                        "data": [{"状态": "失败", "原因": "SQL执行错误，请参考报告文本内容了解详情。"}]
                    })
                
                # 序列化为JSON (使用自定义编码器处理Decimal)
                try:
                    structured_view_json = json.dumps(structured_view, ensure_ascii=False, cls=DecimalEncoder)
                    logger.info(f"序列化成功, JSON长度: {len(structured_view_json)}")
                except Exception as e:
                    logger.error(f"JSON序列化失败: {str(e)}")
                    # 简化结构，重试序列化
                    simple_view = {
                        "vis_type": "vis-dashboard",
                        "vis_content": {
                            "title": "营销活动用户群体分析报告",
                            "display_strategy": "report", 
                            "chart_count": 1,
                            "data": [{
                                "type": "response_table",
                                "title": "数据显示",
                                "describe": "数据序列化失败",
                                "data": [{"错误": "JSON序列化失败，请参考文本报告"}]
                            }]
                        }
                    }
                    structured_view_json = json.dumps(simple_view, ensure_ascii=False)
                
                return ActionOutput(
                    is_exe_success=True,
                    content=full_report,
                    view=structured_view_json,
                    resource_type=self.resource_need.value,
                    resource_value=db._db_name
                )
            
            # 如果没有JSON格式，回退到原始正则匹配
            sql_pattern = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL)
            chart_type_pattern = re.compile(r"display_type:\s*(\w+)")
            
            sql_match = sql_pattern.search(ai_message)
            chart_match = chart_type_pattern.search(ai_message)
            
            if not sql_match:
                return ActionOutput(
                    is_exe_success=True,
                    content=ai_message,
                    view=ai_message,
                )
                
            sql = sql_match.group(1).strip()
            display_type = chart_match.group(1) if chart_match else "response_table"
            
            # 检查并修复SQL
            fixed_sql = check_and_fix_sql(sql)
            if fixed_sql != sql:
                logger.info(f"原SQL: {sql}")
                logger.info(f"修复后: {fixed_sql}")
                sql = fixed_sql
            
            # 检查SQL方言兼容性
            fixed_sql = self.fix_sql_dialect(sql, db.dialect)
            if fixed_sql != sql:
                logger.info(f"SQL方言修复: {sql} -> {fixed_sql}")
                sql = fixed_sql
            
            # 创建参数
            param = SqlInput(
                display_type=display_type,
                sql=sql,
                thought="通过SQL提取的分析结果"
            )
            
            # 构建结构化输出
            formatted_content = f"## 分析结论\n{param.thought}\n\n## 分析图表\n"
            
            # 在执行SQL时添加友好的错误处理
            try:
                # 添加更详细的日志
                logger.info(f"正在执行完整SQL: {fixed_sql}")
                data_df = await db.query_to_df(fixed_sql)
                
                # 使用正确的参数格式调用display方法
                view = await self._render_protocol.display(
                    chart=json.loads(model_to_json(param)), 
                    data_df=data_df
                )
                
                # 构建视图结构
                if isinstance(view, dict):
                    # 转换为dashboard格式
                    dashboard_view = {
                        "vis_type": "vis-dashboard",
                        "vis_content": {
                            "title": "数据分析报告",
                            "display_strategy": "default",
                            "chart_count": 1,
                            "data": [{
                                "type": display_type,
                                "title": "数据分析",
                                "describe": param.thought,
                                "sql": fixed_sql,
                                "data": data_df.to_dict(orient='records') if data_df is not None and not data_df.empty else []
                            }]
                        }
                    }
                    view_json = json.dumps(dashboard_view, ensure_ascii=False, cls=DecimalEncoder)
                else:
                    view_json = view
                
                return ActionOutput(
                    is_exe_success=True,
                    content=formatted_content,
                    view=view_json,
                    resource_type=self.resource_need.value,
                    resource_value=db._db_name,
                )
                
            except Exception as e:
                # 捕获并记录详细错误，包括完整堆栈
                error_full = str(e)
                logger.error(f"SQL执行详细错误: {error_full}", exc_info=True)
                
                # 检查是否涉及MySQL的别名问题
                error_str = error_full.lower()
                error_msg = f"SQL执行失败: {error_full}"
                suggestion = ""
                
                # 针对MySQL特定错误提供清晰的建议
                if "unknown column" in error_str and "in 'group statement'" in error_str:
                    col_match = re.search(r"Unknown column '([^']+)'", error_full)
                    if col_match:
                        problem_col = col_match.group(1)
                        suggestion = f"MySQL不支持在GROUP BY子句中使用列别名。'{problem_col}'似乎是一个别名，请在GROUP BY中使用完整表达式替代。"
                elif "unknown column" in error_str and "in 'having clause'" in error_str:
                    col_match = re.search(r"Unknown column '([^']+)'", error_full)
                    if col_match:
                        problem_col = col_match.group(1)
                        suggestion = f"MySQL不支持在HAVING子句中使用列别名。'{problem_col}'似乎是一个别名，请在HAVING中使用完整表达式替代。"
                elif "unknown column" in error_str and "in 'order clause'" in error_str:
                    col_match = re.search(r"Unknown column '([^']+)'", error_full)
                    if col_match:
                        problem_col = col_match.group(1)
                        suggestion = f"MySQL不支持在ORDER BY子句中使用列别名。'{problem_col}'似乎是一个别名，请在ORDER BY中使用完整表达式替代。"
                elif "syntax" in error_str:
                    suggestion = "SQL语法错误，请检查语法结构是否完整，括号是否配对，关键词是否拼写正确。"
                
                # 返回错误信息但不中断整个流程
                formatted_content += f"### SQL执行错误\n```\n{error_msg}\n```\n\n"
                if suggestion:
                    formatted_content += f"### 修复建议\n```\n{suggestion}\n```\n\n"
                
                # 创建错误视图
                error_view = {
                    "vis_type": "vis-dashboard",
                    "vis_content": {
                        "title": "SQL执行错误",
                        "display_strategy": "report",
                        "chart_count": 1,
                        "data": [{
                            "type": "response_table",
                            "title": "错误信息",
                            "describe": suggestion or "SQL执行失败，请检查语法。",
                            "sql": fixed_sql,
                            "data": [{"错误": error_msg, "建议": suggestion or "请修改SQL语法"}]
                        }]
                    }
                }
                
                # 序列化为JSON
                view_json = json.dumps(error_view, ensure_ascii=False)
                
                return ActionOutput(
                    is_exe_success=True,
                    content=formatted_content,
                    view=view_json,
                    resource_type=self.resource_need.value,
                    resource_value=db._db_name
                )
        except Exception as e:
            logger.exception(f"图表生成失败: {str(e)}")
            return ActionOutput(is_exe_success=True, content=ai_message, view=ai_message)


class SimpleAssistantAgent(ConversableAgent):
    """Simple Assistant Agent with chart capability."""

    profile: ProfileConfig = ProfileConfig(
        name=DynConfig(
            "Tom",
            category="agent",
            key="dbgpt_agent_expand_simple_assistant_agent_profile_name",
        ),
        role=DynConfig(
            "AI Assistant",
            category="agent",
            key="dbgpt_agent_expand_simple_assistant_agent_profile_role",
        ),
        goal=DynConfig(
            "Understand user questions and provide comprehensive data analytics with multi-dimensional visualization",
            category="agent",
            key="dbgpt_agent_expand_simple_assistant_agent_profile_goal",
        ),
        constraints=DynConfig(
            [
                "请确保你的回答清晰、逻辑严谨、友好且易于理解。",
                "进行数据分析时，请从多个维度考虑问题，为每个维度提供独立的SQL查询和可视化图表。",
                "每个分析维度应包含：分析思路、SQL查询和适合的可视化图表。",
                "从以下可视化类型中为每个维度选择最合适的图表类型：{{ display_type }}",
                "图表类型选择指南：柱状图适合分类对比、折线图适合趋势分析、饼图适合占比分析、表格适合详细数据展示、热力图适合多变量关系分析。",
                "封装SQL查询时，使用JSON格式：{\"display_type\": \"图表类型\", \"sql\": \"SQL语句\", \"thought\": \"分析思路\"}",
                "分析必须包含与表结构schema相关的多个维度",
                "务必在分析结论中提供业务洞察和建议",
                "最终输出应为一份结构化报告，包含各维度分析及总体建议",
                "请确保SQL语句: 1. 不包含多余的引号和反斜杠 2. 使用最简单的SQL语法 3. 避免使用复杂的嵌套结构和别名"
            ],
            category="agent",
            key="dbgpt_agent_expand_simple_assistant_agent_profile_constraints",
        ),
        desc=DynConfig(
            "我是一个具有多维度数据分析和可视化能力的AI助手，能够生成全面的分析报告。",
            category="agent",
            key="dbgpt_agent_expand_summary_assistant_agent_profile_desc",
        ),
    )

    def __init__(self, **kwargs):
        """Create a new SimpleAssistantAgent instance."""
        super().__init__(**kwargs)
        self._post_reranks = [RetrieverNameRanker(5)]
        self._init_actions([SimpleChartAction])

    async def load_resource(self, question: str, is_retry_chat: bool = False):
        """Load agent bind resource."""
        if self.resource and self.database:
            try:
                db = self.database
                table_name = "customer_info"  # 根据实际表名调整
                
                # 根据数据库类型选择正确的SQL语法
                columns_sql = ""
                if db.dialect.lower() == "mysql":
                    columns_sql = f"SHOW COLUMNS FROM {table_name}"
                elif db.dialect.lower() == "sqlite":
                    columns_sql = f"PRAGMA table_info({table_name})"
                elif db.dialect.lower() in ["postgresql", "postgres"]:
                    columns_sql = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"
                else:
                    # 默认回退到表的采样查询
                    columns_sql = f"SELECT * FROM {table_name} LIMIT 1"
                    
                logger.info(f"使用SQL获取表结构: {columns_sql}")
                field_names, result = await db.query(columns_sql)
                
                # MySQL结构解析 (SHOW COLUMNS FROM返回的格式)
                schema_info = f"表名: {table_name}\n字段列表:\n"
                if db.dialect.lower() == "mysql":
                    for col in result:
                        col_name = col[0]  # MySQL SHOW COLUMNS返回的字段名在第一列
                        col_type = col[1]  # 数据类型
                        schema_info += f"- {col_name} ({col_type})\n"
                else:
                    # 其他数据库的解析逻辑
                    for col in result:
                        col_name = col[1] if db.dialect.lower() == "sqlite" else col[0]
                        schema_info += f"- {col_name}\n"
                
                # 添加更详细的字段说明和分析示例
                schema_info += """
字段详细说明:
- AcceptedCmp1-5: 客户接受营销活动1-5的次数 (数值)
- Response: 客户对营销活动的响应率 (数值)
- NumWebPurchases: 网站购买次数 (数值)
- NumCatalogPurchases: 目录购买次数 (数值)
- NumStorePurchases: 实体店购买次数 (数值)
- education_*: 教育水平相关字段 (如PhD=博士, Master=硕士)
- Income: 客户年收入 (数值)
- Marital_*: 婚姻状况相关字段
- Kidhome: 家庭中的孩子数量 (数值)
- Teenhome: 家庭中的青少年数量 (数值)

数据分析建议:
1. 营销活动接受度与购买渠道关联分析: 分析不同营销活动接受度与购买渠道的关系
2. 客户细分分析: 基于人口统计信息(收入/教育/婚姻状况)对客户进行分群
3. 消费模式分析: 研究不同客户群体的购买频率和渠道偏好
4. 营销活动效果对比: 比较不同营销活动的响应率和转化效果

建议使用多个维度进行分析，每个维度应包含独立的SQL查询和适合的可视化图表。
"""
                
                db_prompt = f"数据库类型：{db.db_type}，相关表结构定义：\n{schema_info}"
                logger.info(f"获取到表结构: {db_prompt[:500]}...")
                
                # 资源处理
                resource_prompt, resource_reference = await self.resource.get_prompt(
                    lang=self.language, question=question
                )
                
                if resource_prompt:
                    return f"{db_prompt}\n\n{resource_prompt}", resource_reference
                return db_prompt, resource_reference
                
            except Exception as e:
                logger.error(f"获取表结构失败: {str(e)}")
                # 失败后尝试使用原始方法
                schema_info = await self.blocking_func_to_async(
                    self.database.get_schema_link, 
                    db=self.database._db_name, 
                    question=question
                )
                db_prompt = f"数据库类型：{self.database.db_type}，相关表结构定义：{schema_info}"
                resource_prompt, resource_reference = await self.resource.get_prompt(
                    lang=self.language, question=question
                )
                if resource_prompt:
                    return f"{db_prompt}\n\n{resource_prompt}", resource_reference
                return db_prompt, resource_reference
        
        return await super().load_resource(question, is_retry_chat)

    def _init_reply_message(
        self,
        received_message: AgentMessage,
        rely_messages: Optional[List[AgentMessage]] = None,
    ) -> AgentMessage:
        reply_message = super()._init_reply_message(received_message, rely_messages)
        
        # 添加display_type到上下文
        display_types = self.actions[0].render_prompt() if self.actions else None
        
        # 提供更详细的分析指导
        analysis_guide = """
为了提供全面的分析报告，请遵循以下结构:
1. 首先思考用户问题的不同分析维度，至少选择3-5个相关维度进行分析
2. 对每个维度，提供:
   - 清晰的分析思路
   - 准确的SQL查询（确保语法正确且与数据库兼容）
   - 合适的可视化图表类型（根据数据特点选择）
3. 使用JSON格式封装每个维度的分析:
   {"display_type": "图表类型", "sql": "SQL语句", "thought": "分析思路"}
4. 最后提供总体结论和业务建议

请确保SQL查询语法正确，特别注意MySQL不支持在GROUP BY和HAVING中使用列别名。
"""
        
        reply_message.context = {
            "user_question": received_message.content,
            "display_type": display_types,
            "analysis_guide": analysis_guide
        }
        
        return reply_message
        
    @property
    def database(self) -> Optional[DBResource]:
        """获取数据库资源。"""
        if not self.resource:
            return None
        dbs: List[DBResource] = DBResource.from_resource(self.resource)
        if not dbs:
            return None
        return dbs[0]

    def post_filters(self, resource_candidates_map: Optional[Dict[str, Tuple]] = None):
        """Post filters for resource candidates."""
        if resource_candidates_map:
            new_candidates_map = resource_candidates_map.copy()
            filter_hit = False
            for resource, (
                candidates,
                references,
                prompt_template,
            ) in resource_candidates_map.items():
                for rerank in self._post_reranks:
                    filter_candidates = rerank.rank(candidates)
                    new_candidates_map[resource] = [], [], prompt_template
                    if filter_candidates and len(filter_candidates) > 0:
                        new_candidates_map[resource] = (
                            filter_candidates,
                            references,
                            prompt_template,
                        )
                        filter_hit = True
                        break
            if filter_hit:
                logger.info("Post filters hit, use new candidates.")
                return new_candidates_map
        return resource_candidates_map

def get_dimension_name(index):
    """获取维度的名称"""
    dimensions = [
        "营销活动接受次数与购买渠道关联分析", 
        "分营销活动折扣敏感对比", 
        "客户生命周期与营销响应关联趋势",
        "线上购买频次与营销接受率分层分析",
        "用户画像与营销响应分析"
    ]
    return dimensions[index] if index < len(dimensions) else f"维度{index+1}"

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def prepare_sql(sql):
    """预处理SQL，移除多余引号和处理转义字符"""
    # 移除SQL中的转义字符和引号问题
    sql = sql.replace("\\", "").replace("\'", "'").replace('\"', '"')
    # 确保SQL是单行
    sql = re.sub(r'\s+', ' ', sql)
    return sql

def safe_json_sql(sql, display_type, thought):
    """安全构建包含SQL的JSON对象"""
    # 预处理SQL
    clean_sql = prepare_sql(sql)
    # 直接构建字典而非手动JSON字符串
    return {
        "display_type": display_type,
        "sql": clean_sql,
        "thought": thought
    }
