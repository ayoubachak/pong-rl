import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { GameMenuComponent, GameSettings } from './components/game-menu/game-menu.component';
import { PongGameComponent } from './components/pong-game/pong-game.component';
import { AiLearningComponent } from './components/ai-learning/ai-learning.component';

@Component({
  selector: 'app-root',
  imports: [CommonModule, GameMenuComponent, PongGameComponent, AiLearningComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'pong-rl';
  currentView: 'menu' | 'game' | 'ai-learning' = 'menu';
  gameSettings: GameSettings | null = null;

  onStartGame(settings: GameSettings) {
    this.gameSettings = settings;
    this.currentView = 'game';
  }

  onWatchAILearn() {
    this.currentView = 'ai-learning';
  }

  onGameEnded() {
    this.currentView = 'menu';
    this.gameSettings = null;
  }

  onBackToMenu() {
    this.currentView = 'menu';
    this.gameSettings = null;
  }
}
