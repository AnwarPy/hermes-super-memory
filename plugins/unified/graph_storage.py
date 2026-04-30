"""Graph Storage — حفظ وتحميل الرسوم المعرفية

الوظائف:
- حفظ الرسم كـ JSON (node-link format)
- حفظ المجتمعات كـ JSON
- توليد تقرير Markdown تلقائي
- تحميل الرسوم المحفوظة
"""

import json
import networkx as nx
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class GraphStorage:
    """مخزن الرسوم المعرفية"""
    
    def __init__(self, graphs_dir: Optional[str] = None):
        """
        Args:
            graphs_dir: مجلد حفظ الرسوم (افتراضي: ~/.hermes/graphs)
        """
        if graphs_dir:
            self.graphs_dir = Path(graphs_dir).expanduser()
        else:
            self.graphs_dir = Path.home() / ".hermes" / "graphs"
        
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        print(f"مجلد الرسوم: {self.graphs_dir}")
    
    def save(
        self,
        graph: nx.Graph,
        communities: Dict[str, Any],
        project_name: str,
        generate_report: bool = True,
    ) -> Dict[str, Path]:
        """
        حفظ الرسم المعرفي
        
        Args:
            graph: الرسم المعرفي
            communities: نتائج اكتشاف المجتمعات
            project_name: اسم المشروع
            generate_report: توليد تقرير Markdown
        
        Returns:
            مسارات الملفات المحفوظة
        """
        project_dir = self.graphs_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"جاري حفظ مشروع '{project_name}' في {project_dir}")
        
        # 1. حفظ graph.json
        graph_path = project_dir / "graph.json"
        graph_data = nx.node_link_data(graph)
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, separators=(',', ':'))
        print(f"  ✓ graph.json ({self._format_size(graph_path.stat().st_size)})")
        
        # 2. حفظ communities.json
        comm_path = project_dir / "communities.json"
        with open(comm_path, "w", encoding="utf-8") as f:
            json.dump(communities, f, ensure_ascii=False, separators=(',', ':'))
        print(f"  ✓ communities.json ({self._format_size(comm_path.stat().st_size)})")
        
        # 3. توليد تقرير
        report_path = None
        if generate_report:
            report_path = project_dir / "GRAPH_REPORT.md"
            report = self._generate_report(graph, communities, project_name)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"  ✓ GRAPH_REPORT.md ({self._format_size(report_path.stat().st_size)})")
        
        # 4. حفظ metadata
        meta_path = project_dir / "metadata.json"
        metadata = {
            "project_name": project_name,
            "created_at": datetime.now().isoformat(),
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_communities": communities.get("num_communities", 0),
            "modularity": communities.get("modularity", 0),
            "algorithm": communities.get("algorithm", "unknown"),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, separators=(',', ':'))
        print(f"  ✓ metadata.json")
        
        return {
            "graph_json": graph_path,
            "communities_json": comm_path,
            "report_md": report_path,
            "metadata_json": meta_path,
        }
    
    def load(self, project_name: str) -> nx.Graph:
        """
        تحميل رسم معرفي محفوظ
        
        Args:
            project_name: اسم المشروع
        
        Returns:
            الرسم المعرفي
        """
        graph_path = self.graphs_dir / project_name / "graph.json"
        
        if not graph_path.exists():
            raise FileNotFoundError(f"الرسم غير موجود: {graph_path}")
        
        with open(graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
        
        graph = nx.node_link_graph(graph_data)
        # print(f"✓ تم تحميل الرسم '{project_name}': {graph.number_of_nodes()} عقدة، {graph.number_of_edges()} حافة")
        return graph
    
    def load_communities(self, project_name: str) -> Dict[str, Any]:
        """
        تحميل المجتمعات
        
        Args:
            project_name: اسم المشروع
        
        Returns:
            قاموس المجتمعات
        """
        comm_path = self.graphs_dir / project_name / "communities.json"
        
        if not comm_path.exists():
            raise FileNotFoundError(f"المجتمعات غير موجودة: {comm_path}")
        
        with open(comm_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def list_projects(self) -> list:
        """سرد جميع المشاريع المحفوظة"""
        projects = []
        for item in self.graphs_dir.iterdir():
            # تجاهل المجلدات المخفية والنظامية
            if item.name.startswith('.') or not item.is_dir():
                continue
            if (item / "graph.json").exists():
                projects.append(item.name)
        return sorted(projects)
    
    def delete_project(self, project_name: str) -> bool:
        """
        حذف مشروع
        
        Args:
            project_name: اسم المشروع
        
        Returns:
            True إذا تم الحذف
        """
        project_dir = self.graphs_dir / project_name
        
        if not project_dir.exists():
            return False
        
        # حذف جميع الملفات
        for file in project_dir.iterdir():
            file.unlink()
        project_dir.rmdir()
        
        print(f"✓ تم حذف مشروع '{project_name}'")
        return True
    
    def _generate_report(
        self,
        graph: nx.Graph,
        communities: Dict[str, Any],
        project_name: str,
    ) -> str:
        """توليد تقرير Markdown"""
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = nx.density(graph) if num_nodes > 0 else 0
        avg_degree = sum(dict(graph.degree()).values()) / num_nodes if num_nodes > 0 else 0
        
        report = f"""# تقرير الرسم المعرفي: {project_name}

**تاريخ التوليد**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 📊 إحصائيات عامة

| المقياس | القيمة |
|---------|--------|
| عدد العقد | {num_nodes:,} |
| عدد الحواف | {num_edges:,} |
| الكثافة | {density:.6f} |
| متوسط الدرجة | {avg_degree:.2f} |
| عدد المجتمعات | {communities.get('num_communities', 0)} |
| Modularity Score | {communities.get('modularity', 0):.4f} |
| الخوارزمية | {communities.get('algorithm', 'unknown')} |

---

## 🌍 المجتمعات الدلالية

"""
        # إحصائيات المجتمعات
        comm_stats = self._get_community_stats(graph, communities)
        
        for i, stat in enumerate(comm_stats[:10], 1):  # أول 10 مجتمعات
            report += f"### المجتمع #{i} (ID: {stat['community_id']})\n\n"
            report += f"- **العقد**: {stat['num_nodes']}\n"
            report += f"- **الحواف**: {stat['num_edges']}\n"
            report += f"- **الكثافة**: {stat['density']:.4f}\n"
            report += f"- **متوسط الدرجة**: {stat['avg_degree']:.2f}\n\n"
            
            # أول 5 عقد كمثال
            nodes = communities.get('communities', {}).get(stat['community_id'], [])
            if nodes:
                report += "**أول 5 عقد**:\n\n"
                for node in nodes[:5]:
                    content = graph.nodes[node].get('content', '')[:100].replace('\n', ' ')
                    node_type = graph.nodes[node].get('type', 'unknown')
                    report += f"- `{node}` ({node_type}): {content}...\n"
                report += "\n"
        
        if len(comm_stats) > 10:
            report += f"\n*... و{len(comm_stats) - 10} مجتمعات أخرى*\n"
        
        report += f"""
---

## 🔍 ملاحظات

- تم توليد هذا التقرير تلقائياً بواسطة Unified Memory System
- لتحليل مفصل، راجع graph.json و communities.json
- يمكن إعادة اكتشاف المجتمعات بمعاملات مختلفة

---

**Unified Memory System v1.0.0**
"""
        return report
    
    def _get_community_stats(
        self,
        graph: nx.Graph,
        communities: Dict[str, Any],
    ) -> list:
        """إحصائيات المجتمعات"""
        stats = []
        
        for comm_id, nodes in communities.get('communities', {}).items():
            try:
                subgraph = graph.subgraph(nodes)
                stat = {
                    "community_id": comm_id,
                    "num_nodes": len(nodes),
                    "num_edges": subgraph.number_of_edges(),
                    "density": nx.density(subgraph),
                    "avg_degree": sum(dict(subgraph.degree()).values()) / len(nodes) if nodes else 0,
                }
                stats.append(stat)
            except Exception:
                continue
        
        stats.sort(key=lambda x: x["num_nodes"], reverse=True)
        return stats
    
    def _format_size(self, size_bytes: int) -> str:
        """تنسيق حجم الملف"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
