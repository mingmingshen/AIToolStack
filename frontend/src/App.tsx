import React, { useState, useEffect } from 'react';
import { TopNavigation } from './components/TopNavigation';
import { Dashboard } from './components/Dashboard';
import { ProjectSelector } from './components/ProjectSelector';
import { AnnotationWorkbench } from './components/AnnotationWorkbench';
import { TrainingPanel } from './components/TrainingPanel';
import { useTranslation } from 'react-i18next';
import './App.css';

interface Project {
  id: string;
  name: string;
  description: string;
  created_at?: string;
  updated_at?: string;
}

type MenuItem = 'dashboard' | 'projects';

function App() {
  const [activeMenu, setActiveMenu] = useState<MenuItem>('dashboard');
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [showTrainingPanel, setShowTrainingPanel] = useState(false);
  const [trainingProjectId, setTrainingProjectId] = useState<string | null>(null);

  useEffect(() => {
    // 加载项目列表
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000/api'}/projects`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setProjects(data);
    } catch (error) {
      console.error('Failed to fetch projects:', error);
      // 如果后端未启动，显示空列表而不是报错
      setProjects([]);
    }
  };

  const handleProjectSelect = (project: Project) => {
    setSelectedProject(project);
  };

  const handleBackToProjects = () => {
    setSelectedProject(null);
    setActiveMenu('projects'); // 返回到项目管理菜单
    fetchProjects();
  };

  const handleMenuChange = (menu: MenuItem) => {
    setActiveMenu(menu);
    if (menu === 'projects') {
      setSelectedProject(null);
      setShowTrainingPanel(false);
    } else {
      setSelectedProject(null);
      setShowTrainingPanel(false);
    }
  };

  const handleOpenTrainingPanel = (projectId: string) => {
    setTrainingProjectId(projectId);
    setShowTrainingPanel(true);
  };

  const handleCloseTrainingPanel = () => {
    setShowTrainingPanel(false);
    setTrainingProjectId(null);
  };

  // 渲染主要内容
  const renderMainContent = () => {
    // 如果正在显示训练面板，显示训练面板
    if (showTrainingPanel && trainingProjectId) {
      return (
        <TrainingPanel
          projectId={trainingProjectId}
          onClose={handleCloseTrainingPanel}
        />
      );
    }

    // 如果选择了项目，显示标注界面
    if (selectedProject) {
      return (
        <AnnotationWorkbench
          project={selectedProject}
          onBack={handleBackToProjects}
          onOpenTraining={handleOpenTrainingPanel}
        />
      );
    }

    // 根据当前菜单显示不同内容
    switch (activeMenu) {
      case 'dashboard':
        return (
          <Dashboard
            onStartProject={() => {
              setActiveMenu('projects');
              fetchProjects();
            }}
          />
        );

      case 'projects':
    return (
        <ProjectSelector
          projects={projects}
          onSelect={handleProjectSelect}
          onRefresh={fetchProjects}
            onOpenTraining={handleOpenTrainingPanel}
        />
    );

      default:
        return null;
  }
  };

  // 判断当前是否显示标注界面，如果是则移除 padding
  const isAnnotationView = selectedProject !== null;

  return (
    <div className="app">
      <TopNavigation activeMenu={activeMenu} onMenuChange={handleMenuChange} />
      <div 
        className="app-content" 
        style={{ 
          marginTop: '64px',
          padding: isAnnotationView ? '0' : '24px'
        }}
      >
        {renderMainContent()}
      </div>
    </div>
  );
}

export default App;
