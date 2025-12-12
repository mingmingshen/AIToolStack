import React from 'react';
import { useTranslation } from 'react-i18next';
import './ShortcutHelper.css';

interface ShortcutItem {
  keyKey: string;
  descriptionKey: string;
}

export const ShortcutHelper: React.FC = () => {
  const { t } = useTranslation();

  const shortcuts: ShortcutItem[] = [
    { keyKey: 'shortcuts.key1-9', descriptionKey: 'shortcuts.switchClass' },
    { keyKey: 'shortcuts.keyR', descriptionKey: 'shortcuts.bboxTool' },
    { keyKey: 'shortcuts.keyP', descriptionKey: 'shortcuts.polygonTool' },
    { keyKey: 'shortcuts.keyV', descriptionKey: 'shortcuts.selectTool' },
    { keyKey: 'shortcuts.keyK', descriptionKey: 'shortcuts.keypointTool' },
    { keyKey: 'shortcuts.keyPrev', descriptionKey: 'shortcuts.prevImage' },
    { keyKey: 'shortcuts.keyNext', descriptionKey: 'shortcuts.nextImage' },
    { keyKey: 'shortcuts.keyPan', descriptionKey: 'shortcuts.panCanvas' },
    { keyKey: 'shortcuts.keyH', descriptionKey: 'shortcuts.toggleAnnotations' },
    { keyKey: 'shortcuts.keyDelete', descriptionKey: 'shortcuts.deleteAnnotation' },
    { keyKey: 'shortcuts.keyUndo', descriptionKey: 'shortcuts.undo' },
    { keyKey: 'shortcuts.keyRedo', descriptionKey: 'shortcuts.redo' },
    { keyKey: 'shortcuts.keySave', descriptionKey: 'shortcuts.save' },
    { keyKey: 'shortcuts.keyEsc', descriptionKey: 'shortcuts.cancel' },
    { keyKey: 'shortcuts.keyEnter', descriptionKey: 'shortcuts.completePolygon' },
  ];

  return (
    <div className="shortcut-helper">
      <div className="shortcut-helper-content">
        {shortcuts.map((item, index) => (
          <div key={index} className="shortcut-item">
            <span className="shortcut-key">{t(item.keyKey)}</span>
            <span className="shortcut-description">{t(item.descriptionKey)}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
