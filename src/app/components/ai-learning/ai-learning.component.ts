import { Component, ElementRef, ViewChild, OnInit, OnDestroy, AfterViewInit, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as tf from '@tensorflow/tfjs';
import * as d3 from 'd3';

interface AIAgent {
  id: string;
  name: string;
  model: tf.Sequential;
  targetModel: tf.Sequential;
  memory: Experience[];
  epsilon: number;
  score: number;
  totalReward: number;
  gamesPlayed: number;
  color: string;
}

interface Experience {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

interface GameState {
  ballX: number;
  ballY: number;
  ballDx: number;
  ballDy: number;
  paddle1Y: number;
  paddle2Y: number;
  player1Score: number;
  player2Score: number;
}

interface LearningConfig {
  learningRate: number;
  epsilon: number;
  epsilonDecay: number;
  epsilonMin: number;
  batchSize: number;
  memorySize: number;
  targetUpdateFreq: number;
  gameSpeed: number;
  episodes: number;
}

interface TrainingLog {
  timestamp: string;
  gameNumber: number;
  agent1Score: number;
  agent2Score: number;
  agent1Reward: number;
  agent2Reward: number;
  agent1Epsilon: number;
  agent2Epsilon: number;
  message: string;
}

@Component({
  selector: 'app-ai-learning',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './ai-learning.component.html',
  styleUrls: ['./ai-learning.component.css']
})
export class AiLearningComponent implements OnInit, OnDestroy, AfterViewInit {
  @ViewChild('gameCanvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('networkVisualization', { static: true }) networkVisRef!: ElementRef<HTMLDivElement>;
  @Output() backToMenu = new EventEmitter<void>();

  private ctx!: CanvasRenderingContext2D;
  private animationId!: number;
  isTraining = false;
  gameNumber = 0;
  episode = 0;

  // Game configuration
  canvas = { width: 800, height: 400 };
  ball = { x: 400, y: 200, dx: 6, dy: 4, radius: 8 }; // Increased speed from 4,3 to 6,4
  paddle1 = { x: 20, y: 175, width: 10, height: 80 };
  paddle2 = { x: 770, y: 175, width: 10, height: 80 };
  
  // AI Agents
  agent1: AIAgent | null = null;
  agent2: AIAgent | null = null;

  // Training logs and tracking
  trainingLogs: TrainingLog[] = [];
  maxLogs = 100;

  // Configuration for episodes
  config: LearningConfig = {
    learningRate: 0.001,
    epsilon: 1.0,
    epsilonDecay: 0.995,
    epsilonMin: 0.01,
    batchSize: 32,
    memorySize: 10000,
    targetUpdateFreq: 100,
    gameSpeed: 1,
    episodes: 1000
  };

  // Training statistics
  stats = {
    agent1Wins: 0,
    agent2Wins: 0,
    totalGames: 0,
    avgRewardAgent1: 0,
    avgRewardAgent2: 0,
    agent1WinRate: 0,
    agent2WinRate: 0,
    agent1Rewards: [] as number[],
    agent2Rewards: [] as number[]
  };

  // Visualization
  private readonly lossHistory: number[] = [];
  private readonly rewardHistory: { agent1: number[], agent2: number[] } = { agent1: [], agent2: [] };
  private readonly winRateHistory: { agent1: number[], agent2: number[] } = { agent1: [], agent2: [] };

  private readonly networkSvg: any;
  private readonly networkConfig = {
    layers: [8, 128, 64, 32, 3],
    nodeRadius: 8,
    layerSpacing: 120,
    nodeSpacing: 25
  };

  ngOnInit() {
    this.initializeCanvas();
  }

  ngAfterViewInit() {
    this.setupVisualization();
  }

  ngOnDestroy() {
    this.stopTraining();
    this.cleanup();
  }

  private initializeCanvas() {
    const canvas = this.canvasRef.nativeElement;
    canvas.width = this.canvas.width;
    canvas.height = this.canvas.height;
    this.ctx = canvas.getContext('2d')!;
  }

  private async setupVisualization() {
    // Initialize TensorFlow.js visualization
    await tf.ready();
    console.log('TensorFlow.js ready for AI training!');
    
    // Initialize D3.js neural network visualization
    setTimeout(() => {
      this.initializeNetworkVisualization();
    }, 100);
  }

  private createNeuralNetwork(): tf.Sequential {
    const model = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [8], units: 128, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 64, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 32, activation: 'relu' }),
        tf.layers.dense({ units: 3, activation: 'linear' }) // 3 actions: up, stay, down
      ]
    });

    model.compile({
      optimizer: tf.train.adam(this.config.learningRate),
      loss: 'meanSquaredError'
    });

    return model;
  }

  // Enhanced training loop with logging
  private async trainingLoop() {
    if (!this.isTraining || !this.agent1 || !this.agent2) return;

    // Game simulation step
    await this.gameStep();

    // Log significant events
    if (this.gameNumber % 50 === 0 && this.gameNumber > 0) {
      this.addTrainingLog(`Milestone: ${this.gameNumber} games completed`);
    }

    // Update visualization every 10 games
    if (this.gameNumber % 10 === 0) {
      this.updateVisualization();
    }

    // Continue training
    this.animationId = requestAnimationFrame(() => this.trainingLoop());
  }

  // Enhanced start training with better logging
  async startTraining() {
    if (this.isTraining) return;

    this.isTraining = true;
    this.gameNumber = 0;
    this.episode = 0;

    this.addTrainingLog('🚀 Training session started');
    this.addTrainingLog('🧠 Initializing neural networks...');

    // Initialize AI agents
    this.agent1 = {
      id: 'agent1',
      name: 'Red AI',
      model: this.createNeuralNetwork(),
      targetModel: this.createNeuralNetwork(),
      memory: [],
      epsilon: this.config.epsilon,
      score: 0,
      totalReward: 0,
      gamesPlayed: 0,
      color: '#ff4444'
    };

    this.agent2 = {
      id: 'agent2',
      name: 'Blue AI',
      model: this.createNeuralNetwork(),
      targetModel: this.createNeuralNetwork(),
      memory: [],
      epsilon: this.config.epsilon,
      score: 0,
      totalReward: 0,
      gamesPlayed: 0,
      color: '#4444ff'
    };

    // Copy initial weights to target networks
    this.agent1.targetModel.setWeights(this.agent1.model.getWeights());
    this.agent2.targetModel.setWeights(this.agent2.model.getWeights());

    this.addTrainingLog('✅ Agents initialized successfully');
    this.addTrainingLog('🎮 Starting game simulation...');

    this.resetGame();
    this.trainingLoop();
  }

  // Enhanced stop training with logging
  stopTraining() {
    this.isTraining = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    
    if (this.gameNumber > 0) {
      this.addTrainingLog(`⏹️ Training stopped after ${this.gameNumber} games`);
      const agent1WinRate = this.stats.agent1WinRate.toFixed(1);
      const agent2WinRate = this.stats.agent2WinRate.toFixed(1);
      this.addTrainingLog(`📊 Final win rates - Red: ${agent1WinRate}%, Blue: ${agent2WinRate}%`);
    }
  }

  private async gameStep() {
    if (!this.agent1 || !this.agent2) return;

    // Get current state
    const state = this.getGameState();

    // Get actions from both agents
    const action1 = await this.getAction(this.agent1, state);
    const action2 = await this.getAction(this.agent2, state);

    // Store previous state
    const prevState = [...state];

    // Execute actions
    this.executeAction(1, action1);
    this.executeAction(2, action2);

    // Update game physics
    this.updateBall();
    const gameResult = this.checkGameEnd();

    // Get new state
    const newState = this.getGameState();

    // Calculate rewards
    const rewards = this.calculateRewards(gameResult);

    // Store experiences
    this.storeExperience(this.agent1, prevState, action1, rewards.agent1, newState, gameResult.gameEnded);
    this.storeExperience(this.agent2, prevState, action2, rewards.agent2, newState, gameResult.gameEnded);

    // Train networks
    if (this.agent1.memory.length >= this.config.batchSize) {
      await this.trainAgent(this.agent1);
    }
    if (this.agent2.memory.length >= this.config.batchSize) {
      await this.trainAgent(this.agent2);
    }

    // Update target networks periodically
    this.episode++;
    if (this.episode % this.config.targetUpdateFreq === 0) {
      this.updateTargetNetworks();
    }

    // Handle game end
    if (gameResult.gameEnded) {
      this.handleGameEnd(gameResult.winner);
      this.resetGame();
      this.gameNumber++;
    }

    // Render game
    this.render();
  }

  private getGameState(): number[] {
    // Normalize all values to [-1, 1] range
    return [
      (this.ball.x / this.canvas.width) * 2 - 1,
      (this.ball.y / this.canvas.height) * 2 - 1,
      this.ball.dx / 10,
      this.ball.dy / 10,
      (this.paddle1.y / this.canvas.height) * 2 - 1,
      (this.paddle2.y / this.canvas.height) * 2 - 1,
      (this.ball.x - this.paddle1.x) / this.canvas.width,
      (this.ball.x - this.paddle2.x) / this.canvas.width
    ];
  }

  private lastAction = 1; // Default to "stay"
  private getLastAction(): number {
    return this.lastAction;
  }

  private async getAction(agent: AIAgent, state: number[]): Promise<number> {
    // Epsilon-greedy action selection
    if (Math.random() < agent.epsilon) {
      this.lastAction = Math.floor(Math.random() * 3);
      return this.lastAction;
    }

    // Predict Q-values
    const stateTensor = tf.tensor2d([state]);
    const qValues = agent.model.predict(stateTensor) as tf.Tensor;
    const qArray = await qValues.data();
    stateTensor.dispose();
    qValues.dispose();

    // Return action with highest Q-value
    this.lastAction = qArray.indexOf(Math.max(...Array.from(qArray)));
    return this.lastAction;
  }

  private executeAction(player: number, action: number) {
    const paddle = player === 1 ? this.paddle1 : this.paddle2;
    const speed = 8;

    switch (action) {
      case 0: // Move up
        paddle.y = Math.max(0, paddle.y - speed);
        break;
      case 1: // Stay
        break;
      case 2: // Move down
        paddle.y = Math.min(this.canvas.height - paddle.height, paddle.y + speed);
        break;
    }
  }

  private updateBall() {
    this.ball.x += this.ball.dx * this.config.gameSpeed;
    this.ball.y += this.ball.dy * this.config.gameSpeed;

    // Ball collision with top and bottom walls
    if (this.ball.y - this.ball.radius <= 0 || this.ball.y + this.ball.radius >= this.canvas.height) {
      this.ball.dy = -this.ball.dy;
    }

    // Ball collision with paddles
    if (this.ballCollidesWithPaddle(this.paddle1) || this.ballCollidesWithPaddle(this.paddle2)) {
      this.ball.dx = -this.ball.dx;
      // Add some randomness to make it more interesting
      this.ball.dy += (Math.random() - 0.5) * 2;
    }
  }

  private ballCollidesWithPaddle(paddle: any): boolean {
    return this.ball.x - this.ball.radius <= paddle.x + paddle.width &&
           this.ball.x + this.ball.radius >= paddle.x &&
           this.ball.y - this.ball.radius <= paddle.y + paddle.height &&
           this.ball.y + this.ball.radius >= paddle.y;
  }

  private checkGameEnd(): { gameEnded: boolean, winner: number | null } {
    if (this.ball.x < 0) {
      return { gameEnded: true, winner: 2 }; // Agent 2 wins
    } else if (this.ball.x > this.canvas.width) {
      return { gameEnded: true, winner: 1 }; // Agent 1 wins
    }
    return { gameEnded: false, winner: null };
  }

  private calculateRewards(gameResult: { gameEnded: boolean, winner: number | null }) {
    let agent1Reward = 0;
    let agent2Reward = 0;

    if (gameResult.gameEnded) {
      if (gameResult.winner === 1) {
        agent1Reward = 10; // Win reward
        agent2Reward = -10; // Loss penalty
      } else if (gameResult.winner === 2) {
        agent1Reward = -10; // Loss penalty
        agent2Reward = 10; // Win reward
      }
    } else {
      // Small rewards for keeping the ball in play
      const ballDistanceFromCenter = Math.abs(this.ball.y - this.canvas.height / 2);
      const normalizedDistance = ballDistanceFromCenter / (this.canvas.height / 2);
      agent1Reward = 0.1 * (1 - normalizedDistance);
      agent2Reward = 0.1 * (1 - normalizedDistance);
    }

    return { agent1: agent1Reward, agent2: agent2Reward };
  }

  private storeExperience(agent: AIAgent, state: number[], action: number, reward: number, nextState: number[], done: boolean) {
    agent.memory.push({ state, action, reward, nextState, done });
    agent.totalReward += reward;

    // Limit memory size
    if (agent.memory.length > this.config.memorySize) {
      agent.memory.shift();
    }
  }

  private async trainAgent(agent: AIAgent) {
    if (agent.memory.length < this.config.batchSize) return;

    // Sample random batch
    const batch = this.sampleBatch(agent.memory, this.config.batchSize);
    
    const states = batch.map(exp => exp.state);
    const nextStates = batch.map(exp => exp.nextState);

    // Get current Q-values
    const currentQs = agent.model.predict(tf.tensor2d(states)) as tf.Tensor;
    
    // Get next Q-values from target network
    const nextQs = agent.targetModel.predict(tf.tensor2d(nextStates)) as tf.Tensor;
    
    const currentQsArray = await currentQs.data();
    const nextQsArray = await nextQs.data();

    // Prepare training data
    const trainX: number[][] = [];
    const trainY: number[][] = [];

    for (let i = 0; i < batch.length; i++) {
      const exp = batch[i];
      const currentQ = Array.from(currentQsArray.slice(i * 3, (i + 1) * 3));
      const nextQ = Array.from(nextQsArray.slice(i * 3, (i + 1) * 3));
      
      const targetQ = exp.reward;
      if (!exp.done) {
        // Q-learning update rule
        currentQ[exp.action] = targetQ + 0.95 * Math.max(...nextQ);
      } else {
        currentQ[exp.action] = targetQ;
      }
      
      trainX.push(exp.state);
      trainY.push(currentQ);
    }

    // Train the model
    const xs = tf.tensor2d(trainX);
    const ys = tf.tensor2d(trainY);
    
    await agent.model.fit(xs, ys, { epochs: 1, verbose: 0 });

    // Cleanup tensors
    currentQs.dispose();
    nextQs.dispose();
    xs.dispose();
    ys.dispose();

    // Decay epsilon
    agent.epsilon = Math.max(this.config.epsilonMin, agent.epsilon * this.config.epsilonDecay);
  }

  private sampleBatch(memory: Experience[], batchSize: number): Experience[] {
    const batch: Experience[] = [];
    for (let i = 0; i < batchSize; i++) {
      const randomIndex = Math.floor(Math.random() * memory.length);
      batch.push(memory[randomIndex]);
    }
    return batch;
  }

  private updateTargetNetworks() {
    if (this.agent1 && this.agent2) {
      this.agent1.targetModel.setWeights(this.agent1.model.getWeights());
      this.agent2.targetModel.setWeights(this.agent2.model.getWeights());
    }
  }

  // Enhanced game end handling with logging
  private handleGameEnd(winner: number | null) {
    if (!this.agent1 || !this.agent2) return;

    this.agent1.gamesPlayed++;
    this.agent2.gamesPlayed++;

    let logMessage = '';
    if (winner === 1) {
      this.agent1.score++;
      this.stats.agent1Wins++;
      logMessage = `🔴 Red AI wins game ${this.gameNumber + 1}`;
    } else if (winner === 2) {
      this.agent2.score++;
      this.stats.agent2Wins++;
      logMessage = `🔵 Blue AI wins game ${this.gameNumber + 1}`;
    } else {
      logMessage = `⚽ Game ${this.gameNumber + 1} ended in a draw`;
    }

    this.stats.totalGames++;
    this.updateStats();

    // Log game results periodically
    if (this.gameNumber % 25 === 0) {
      this.addTrainingLog(logMessage);
    }

    // Log epsilon decay milestones
    if (this.agent1.epsilon <= 0.5 && this.agent1.epsilon > 0.49) {
      this.addTrainingLog('🎯 Exploration reduced to 50% - agents becoming more strategic');
    }
    if (this.agent1.epsilon <= 0.1 && this.agent1.epsilon > 0.09) {
      this.addTrainingLog('🧠 Exploration at 10% - agents mostly exploiting learned strategies');
    }
  }

  // Add training log entry
  private addTrainingLog(message: string, gameData?: any) {
    const log: TrainingLog = {
      timestamp: new Date().toLocaleTimeString(),
      gameNumber: this.gameNumber,
      agent1Score: this.agent1?.score ?? 0,
      agent2Score: this.agent2?.score ?? 0,
      agent1Reward: this.agent1?.totalReward ?? 0,
      agent2Reward: this.agent2?.totalReward ?? 0,
      agent1Epsilon: this.agent1?.epsilon ?? 0,
      agent2Epsilon: this.agent2?.epsilon ?? 0,
      message: message
    };

    this.trainingLogs.unshift(log);
    
    // Keep only the last maxLogs entries
    if (this.trainingLogs.length > this.maxLogs) {
      this.trainingLogs = this.trainingLogs.slice(0, this.maxLogs);
    }
  }

  // Navigate back to menu
  goBackToMenu() {
    this.stopTraining();
    this.backToMenu.emit();
  }

  // TrackBy function for ngFor performance
  trackByIndex(index: number, item: any): number {
    return index;
  }

  // Configuration update methods
  updateLearningRate(value: number) {
    this.config.learningRate = value;
    this.addTrainingLog(`Learning rate updated to ${value.toFixed(4)}`);
    if (this.agent1 && this.agent2) {
      // Recompile models with new learning rate
      this.agent1.model.compile({
        optimizer: tf.train.adam(this.config.learningRate),
        loss: 'meanSquaredError'
      });
      this.agent2.model.compile({
        optimizer: tf.train.adam(this.config.learningRate),
        loss: 'meanSquaredError'
      });
    }
  }

  updateGameSpeed(value: number) {
    this.config.gameSpeed = value;
    this.addTrainingLog(`Game speed updated to ${value}x`);
  }

  updateEpsilonDecay(value: number) {
    this.config.epsilonDecay = value;
    this.addTrainingLog(`Epsilon decay updated to ${value.toFixed(4)}`);
  }

  // Advanced visualization methods
  private createPerformanceChart() {
    const chartContainer = d3.select(this.networkVisRef.nativeElement.parentElement)
      .append('div')
      .attr('class', 'performance-chart')
      .style('margin-top', '20px');

    const width = 600;
    const height = 200;

    const svg = chartContainer.append('svg')
      .attr('width', width)
      .attr('height', height)
      .style('background', 'rgba(15, 15, 35, 0.8)')
      .style('border', '1px solid #00ff00')
      .style('border-radius', '10px');

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, 100])
      .range([40, width - 40]);

    const yScale = d3.scaleLinear()
      .domain([0, 100])
      .range([height - 40, 40]);

    // Add axes
    svg.append('g')
      .attr('transform', `translate(0, ${height - 40})`)
      .call(d3.axisBottom(xScale))
      .selectAll('text')
      .style('fill', '#00ff00')
      .style('font-family', 'monospace');

    svg.append('g')
      .attr('transform', 'translate(40, 0)')
      .call(d3.axisLeft(yScale))
      .selectAll('text')
      .style('fill', '#00ff00')
      .style('font-family', 'monospace');

    // Add grid lines - Fixed tickFormat to use null instead of empty string
    svg.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0, ${height - 40})`)
      .call(d3.axisBottom(xScale).tickSize(-height + 80).tickFormat(null))
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.3);

    svg.append('g')
      .attr('class', 'grid')
      .attr('transform', 'translate(40, 0)')
      .call(d3.axisLeft(yScale).tickSize(-width + 80).tickFormat(null))
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.3);

    // Labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 5)
      .attr('text-anchor', 'middle')
      .style('fill', '#00ff00')
      .style('font-family', 'monospace')
      .style('font-size', '12px')
      .text('Games Played');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .style('fill', '#00ff00')
      .style('font-family', 'monospace')
      .style('font-size', '12px')
      .text('Win Rate %');

    return svg;
  }

  // Export/Import model functionality
  async exportModel(agentType: 'agent1' | 'agent2') {
    const agent = agentType === 'agent1' ? this.agent1 : this.agent2;
    if (!agent) return;

    try {
      // Create a downloadable model file
      const modelData = {
        modelWeights: agent.model.getWeights().map(w => w.arraySync()),
        config: {
          epsilon: agent.epsilon,
          totalReward: agent.totalReward,
          gamesPlayed: agent.gamesPlayed,
          score: agent.score
        },
        exportDate: new Date().toISOString(),
        agentType: agentType
      };

      const blob = new Blob([JSON.stringify(modelData, null, 2)], {
        type: 'application/json'
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `pong-ai-${agentType}-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      this.addTrainingLog(`${agentType} model exported successfully`);
    } catch (error) {
      console.error('Error exporting model:', error);
      this.addTrainingLog(`Error exporting ${agentType} model`);
    }
  }

  async importModel(agentType: 'agent1' | 'agent2', file: File) {
    if (!file) return;

    try {
      this.addTrainingLog(`📥 Importing model for ${agentType}...`);
      
      const text = await file.text();
      const modelData = JSON.parse(text);

      const agent = agentType === 'agent1' ? this.agent1 : this.agent2;
      if (!agent) {
        this.addTrainingLog(`❌ Cannot import to ${agentType} - agent not initialized`);
        return;
      }

      // Restore weights
      const weights = modelData.modelWeights.map((w: any) => tf.tensor(w));
      agent.model.setWeights(weights);
      agent.targetModel.setWeights(weights);

      // Restore stats if available
      if (modelData.config) {
        agent.epsilon = modelData.config.epsilon ?? agent.epsilon;
        agent.totalReward = modelData.config.totalReward ?? 0;
        agent.gamesPlayed = modelData.config.gamesPlayed ?? 0;
        agent.score = modelData.config.score ?? 0;
      }

      // Cleanup tensors
      weights.forEach((w: tf.Tensor) => w.dispose());

      this.addTrainingLog(`✅ Model imported successfully for ${agent.name}`);
      this.addTrainingLog(`📊 Restored stats: ${agent.gamesPlayed} games, ${agent.score} wins`);
    } catch (error) {
      console.error('Error importing model:', error);
      this.addTrainingLog(`❌ Failed to import model for ${agentType}`);
    }
  }

  // Reset training progress
  resetTraining() {
    this.stopTraining();
    
    this.addTrainingLog('🔄 Resetting training session...');
    
    // Reset statistics
    this.stats = {
      agent1Wins: 0,
      agent2Wins: 0,
      totalGames: 0,
      avgRewardAgent1: 0,
      avgRewardAgent2: 0,
      agent1WinRate: 0,
      agent2WinRate: 0,
      agent1Rewards: [] as number[],
      agent2Rewards: [] as number[]
    };

    // Clear history
    this.rewardHistory.agent1.length = 0;
    this.rewardHistory.agent2.length = 0;
    this.winRateHistory.agent1.length = 0;
    this.winRateHistory.agent2.length = 0;
    this.lossHistory.length = 0;

    // Reset game counter
    this.gameNumber = 0;
    this.episode = 0;

    // Dispose and recreate agents if they exist
    if (this.agent1) {
      this.agent1.model.dispose();
      this.agent1.targetModel.dispose();
      this.agent1 = null;
    }
    
    if (this.agent2) {
      this.agent2.model.dispose();
      this.agent2.targetModel.dispose();
      this.agent2 = null;
    }

    // Clear visualization
    if (this.networkVisRef?.nativeElement) {
      d3.select(this.networkVisRef.nativeElement).selectAll('*').remove();
    }

    this.addTrainingLog('✅ Training session reset complete');
    this.addTrainingLog('🎯 Ready to start new training session');
  }

  // Clean up resources
  private cleanup() {
    if (this.agent1) {
      this.agent1.model.dispose();
      this.agent1.targetModel.dispose();
    }
    
    if (this.agent2) {
      this.agent2.model.dispose();
      this.agent2.targetModel.dispose();
    }

    // Clear D3 visualization
    if (this.networkVisRef?.nativeElement) {
      d3.select(this.networkVisRef.nativeElement).selectAll('*').remove();
    }
  }

  private updateStats() {
    if (this.stats.totalGames > 0) {
      this.stats.agent1WinRate = (this.stats.agent1Wins / this.stats.totalGames) * 100;
      this.stats.agent2WinRate = (this.stats.agent2Wins / this.stats.totalGames) * 100;
    }

    if (this.agent1 && this.agent2) {
      this.stats.avgRewardAgent1 = this.agent1.totalReward / Math.max(1, this.agent1.gamesPlayed);
      this.stats.avgRewardAgent2 = this.agent2.totalReward / Math.max(1, this.agent2.gamesPlayed);
    }
  }

  private resetGame() {
    this.ball.x = this.canvas.width / 2;
    this.ball.y = this.canvas.height / 2;
    this.ball.dx = (Math.random() > 0.5 ? 1 : -1) * 4;
    this.ball.dy = (Math.random() - 0.5) * 6;
    
    this.paddle1.y = (this.canvas.height - this.paddle1.height) / 2;
    this.paddle2.y = (this.canvas.height - this.paddle2.height) / 2;
  }

  private updateVisualization() {
    // Update reward history
    if (this.agent1 && this.agent2) {
      this.rewardHistory.agent1.push(this.agent1.totalReward);
      this.rewardHistory.agent2.push(this.agent2.totalReward);
      
      this.winRateHistory.agent1.push(this.stats.agent1WinRate);
      this.winRateHistory.agent2.push(this.stats.agent2WinRate);
      
      // Keep only last 100 data points
      if (this.rewardHistory.agent1.length > 100) {
        this.rewardHistory.agent1.shift();
        this.rewardHistory.agent2.shift();
        this.winRateHistory.agent1.shift();
        this.winRateHistory.agent2.shift();
      }
    }
  }

  private render() {
    // Clear canvas
    this.ctx.fillStyle = '#0f0f23';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw center line
    this.ctx.strokeStyle = '#00ff00';
    this.ctx.setLineDash([5, 5]);
    this.ctx.beginPath();
    this.ctx.moveTo(this.canvas.width / 2, 0);
    this.ctx.lineTo(this.canvas.width / 2, this.canvas.height);
    this.ctx.stroke();
    this.ctx.setLineDash([]);

    // Draw paddles with AI agent colors
    if (this.agent1) {
      this.ctx.fillStyle = this.agent1.color;
      this.ctx.fillRect(this.paddle1.x, this.paddle1.y, this.paddle1.width, this.paddle1.height);
    }

    if (this.agent2) {
      this.ctx.fillStyle = this.agent2.color;
      this.ctx.fillRect(this.paddle2.x, this.paddle2.y, this.paddle2.width, this.paddle2.height);
    }

    // Draw ball
    this.ctx.fillStyle = '#00ff00';
    this.ctx.beginPath();
    this.ctx.arc(this.ball.x, this.ball.y, this.ball.radius, 0, Math.PI * 2);
    this.ctx.fill();

    // Update neural network visualization
    this.updateNetworkVisualization();
  }

  private initializeNetworkVisualization() {
    if (!this.networkVisRef?.nativeElement) return;

    const container = d3.select(this.networkVisRef.nativeElement);
    container.selectAll('*').remove();

    // Create two network visualizations side by side
    const containerWidth = 1000;
    const containerHeight = 400;
    const networkWidth = 450;
    const networkHeight = 350;

    // Main container
    const mainSvg = container.append('svg')
      .attr('width', containerWidth)
      .attr('height', containerHeight)
      .style('background', 'rgba(15, 15, 35, 0.8)')
      .style('border', '2px solid #00ff00')
      .style('border-radius', '15px');

    // Agent 1 Network (Left side)
    const agent1Group = mainSvg.append('g')
      .attr('transform', 'translate(25, 25)');

    this.createAgentNetwork(agent1Group, networkWidth, networkHeight, 'agent1', '#ff4444');

    // Agent 2 Network (Right side)
    const agent2Group = mainSvg.append('g')
      .attr('transform', `translate(${25 + networkWidth + 50}, 25)`);

    this.createAgentNetwork(agent2Group, networkWidth, networkHeight, 'agent2', '#4444ff');

    // Add title
    mainSvg.append('text')
      .attr('x', containerWidth / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .style('fill', '#00ff00')
      .style('font-family', 'monospace')
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .text('🧠 REAL-TIME NEURAL NETWORK VISUALIZATION');
  }

  private createAgentNetwork(container: any, width: number, height: number, agentId: string, color: string) {
    const layers = [8, 32, 16, 3]; // Simplified for better visualization
    const nodeRadius = 6;
    const layerSpacing = width / (layers.length + 1);
    
    // Agent title
    container.append('text')
      .attr('x', width / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .style('fill', color)
      .style('font-family', 'monospace')
      .style('font-size', '14px')
      .style('font-weight', 'bold')
      .text(`${agentId === 'agent1' ? 'RED AI' : 'BLUE AI'} BRAIN`);

    // Create nodes and links
    const nodes: any[] = [];
    const links: any[] = [];

    // Generate nodes
    for (let layerIndex = 0; layerIndex < layers.length; layerIndex++) {
      const layerSize = layerIndex === 1 ? 8 : layers[layerIndex]; // Limit middle layers for visualization
      const x = (layerIndex + 1) * layerSpacing;
      const startY = (height - (layerSize - 1) * 25) / 2;

      for (let nodeIndex = 0; nodeIndex < layerSize; nodeIndex++) {
        const y = startY + nodeIndex * 25;
        
        nodes.push({
          id: `${agentId}-${layerIndex}-${nodeIndex}`,
          x: x,
          y: y,
          layer: layerIndex,
          index: nodeIndex,
          agentId: agentId,
          activation: 0
        });

        // Create links to next layer
        if (layerIndex < layers.length - 1) {
          const nextLayerSize = layerIndex + 1 === 1 ? 8 : layers[layerIndex + 1];
          for (let nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++) {
            links.push({
              id: `${agentId}-link-${layerIndex}-${nodeIndex}-${nextNodeIndex}`,
              source: `${agentId}-${layerIndex}-${nodeIndex}`,
              target: `${agentId}-${layerIndex + 1}-${nextNodeIndex}`,
              weight: (Math.random() - 0.5) * 2,
              agentId: agentId
            });
          }
        }
      }
    }

    // Draw connections first (so they appear behind nodes)
    container.selectAll('.network-link')
      .data(links)
      .enter()
      .append('line')
      .attr('class', `network-link ${agentId}-link`)
      .attr('x1', (d: any) => nodes.find(n => n.id === d.source)?.x ?? 0)
      .attr('y1', (d: any) => nodes.find(n => n.id === d.source)?.y ?? 0)
      .attr('x2', (d: any) => nodes.find(n => n.id === d.target)?.x ?? 0)
      .attr('y2', (d: any) => nodes.find(n => n.id === d.target)?.y ?? 0)
      .attr('stroke', '#666')
      .attr('stroke-width', 1)
      .attr('stroke-opacity', 0.2);

    // Draw nodes
    container.selectAll('.network-node')
      .data(nodes)
      .enter()
      .append('circle')
      .attr('class', `network-node ${agentId}-node`)
      .attr('cx', (d: any) => d.x)
      .attr('cy', (d: any) => d.y)
      .attr('r', nodeRadius)
      .attr('fill', '#333')
      .attr('stroke', color)
      .attr('stroke-width', 1);

    // Add input labels
    const inputLabels = ['Ball X', 'Ball Y', 'Ball ΔX', 'Ball ΔY', 'My Paddle', 'Enemy Paddle', 'Distance X', 'Distance Y'];
    nodes.filter(n => n.layer === 0).forEach((node, i) => {
      container.append('text')
        .attr('x', node.x - 45)
        .attr('y', node.y + 3)
        .attr('text-anchor', 'end')
        .style('fill', '#ccc')
        .style('font-family', 'monospace')
        .style('font-size', '8px')
        .text(inputLabels[i] || `Input ${i + 1}`);
    });

    // Add output labels
    const outputLabels = ['Move Up', 'Stay', 'Move Down'];
    nodes.filter(n => n.layer === layers.length - 1).forEach((node, i) => {
      container.append('text')
        .attr('x', node.x + 15)
        .attr('y', node.y + 3)
        .attr('text-anchor', 'start')
        .style('fill', '#ccc')
        .style('font-family', 'monospace')
        .style('font-size', '8px')
        .text(outputLabels[i] || `Output ${i + 1}`);
    });

    // Add layer labels
    const layerLabels = ['INPUTS', 'HIDDEN', 'HIDDEN', 'ACTIONS'];
    for (let i = 0; i < layers.length; i++) {
      container.append('text')
        .attr('x', (i + 1) * layerSpacing)
        .attr('y', height - 10)
        .attr('text-anchor', 'middle')
        .style('fill', color)
        .style('font-family', 'monospace')
        .style('font-size', '10px')
        .style('font-weight', 'bold')
        .text(layerLabels[i]);
    }

    // Add activation indicator
    container.append('text')
      .attr('class', `${agentId}-activation-text`)
      .attr('x', width / 2)
      .attr('y', height + 15)
      .attr('text-anchor', 'middle')
      .style('fill', color)
      .style('font-family', 'monospace')
      .style('font-size', '10px')
      .text('Thinking...');
  }

  private async updateNetworkVisualization() {
    if (!this.networkSvg && !this.agent1 && !this.agent2) return;

    const state = this.getGameState();

    // Get actual predictions from both agents if they exist
    let agent1Predictions: number[] = [];
    let agent2Predictions: number[] = [];

    try {
      if (this.agent1) {
        const stateTensor = tf.tensor2d([state]);
        const predictions = this.agent1.model.predict(stateTensor) as tf.Tensor;
        agent1Predictions = Array.from(await predictions.data());
        stateTensor.dispose();
        predictions.dispose();
      }

      if (this.agent2) {
        const stateTensor = tf.tensor2d([state]);
        const predictions = this.agent2.model.predict(stateTensor) as tf.Tensor;
        agent2Predictions = Array.from(await predictions.data());
        stateTensor.dispose();
        predictions.dispose();
      }
    } catch (error) {
      console.warn('Error getting network predictions:', error);
    }

    // Update Agent 1 network
    this.updateAgentVisualization('agent1', state, agent1Predictions, '#ff4444');
    
    // Update Agent 2 network  
    this.updateAgentVisualization('agent2', state, agent2Predictions, '#4444ff');
  }

  private updateAgentVisualization(agentId: string, state: number[], predictions: number[], color: string) {
    const svg = d3.select(this.networkVisRef.nativeElement).select('svg');

    // Update input layer with actual game state
    svg.selectAll(`.${agentId}-node`)
      .filter((d: any) => d.layer === 0)
      .transition()
      .duration(100)
      .attr('fill', (d: any) => {
        const activation = Math.abs(state[d.index] || 0);
        const intensity = Math.min(activation, 1);
        return d3.interpolateRgb('#000', color)(intensity * 0.8 + 0.2);
      })
      .attr('r', (d: any) => {
        const activation = Math.abs(state[d.index] || 0);
        return 6 + activation * 3; // Pulse based on activation
      });

    // Update hidden layers with simulated activations based on input
    svg.selectAll(`.${agentId}-node`)
      .filter((d: any) => d.layer > 0 && d.layer < 3)
      .transition()
      .duration(150)
      .attr('fill', (d: any) => {
        // Simulate realistic hidden layer activations
        const baseActivation = state.reduce((sum, val, i) => sum + Math.abs(val) * Math.sin(i + d.index), 0) / state.length;
        const activation = Math.abs(Math.sin(baseActivation + Date.now() / 1000 + d.index)) * 0.7 + 0.3;
        return d3.interpolateRgb('#000', color)(activation);
      })
      .attr('r', 6);

    // Update output layer with actual predictions
    if (predictions.length >= 3) {
      const maxPrediction = Math.max(...predictions);
      const chosenAction = predictions.indexOf(maxPrediction);

      svg.selectAll(`.${agentId}-node`)
        .filter((d: any) => d.layer === 3)
        .transition()
        .duration(100)
        .attr('fill', (d: any) => {
          const prediction = predictions[d.index] || 0;
          const normalized = (prediction + 10) / 20; // Normalize Q-values roughly to 0-1
          const intensity = Math.max(0, Math.min(1, normalized));
          return d3.interpolateRgb('#000', color)(intensity);
        })
        .attr('r', (d: any) => {
          const isChosen = d.index === chosenAction;
          return isChosen ? 10 : 6; // Highlight chosen action
        })
        .attr('stroke-width', (d: any) => {
          const isChosen = d.index === chosenAction;
          return isChosen ? 3 : 1;
        });

      // Update action text
      const actionNames = ['Moving Up', 'Staying', 'Moving Down'];
      svg.select(`.${agentId}-activation-text`)
        .text(`Decision: ${actionNames[chosenAction]} (Q: ${maxPrediction.toFixed(2)})`);
    }

    // Update connection weights to show information flow
    svg.selectAll(`.${agentId}-link`)
      .transition()
      .duration(200)
      .attr('stroke-opacity', () => 0.1 + Math.random() * 0.3)
      .attr('stroke-width', () => 0.5 + Math.random() * 1.5)
      .attr('stroke', () => {
        const intensity = Math.random();
        return intensity > 0.7 ? color : '#666';
      });
  }

  getAgent1Stats() {
    if (!this.agent1) return null;
    
    return {
      name: 'RED AI',
      wins: this.stats.agent1Wins,
      winRate: this.stats.totalGames > 0 ? ((this.stats.agent1Wins / this.stats.totalGames) * 100).toFixed(1) : '0.0',
      avgReward: this.stats.agent1Rewards.length > 0 ? 
        (this.stats.agent1Rewards.reduce((a, b) => a + b, 0) / this.stats.agent1Rewards.length).toFixed(2) : '0.00',
      epsilon: this.agent1.epsilon.toFixed(3),
      gamesPlayed: this.stats.totalGames
    };
  }

  getAgent2Stats() {
    if (!this.agent2) return null;
    
    return {
      name: 'BLUE AI',
      wins: this.stats.agent2Wins,
      winRate: this.stats.totalGames > 0 ? ((this.stats.agent2Wins / this.stats.totalGames) * 100).toFixed(1) : '0.0',
      avgReward: this.stats.agent2Rewards.length > 0 ? 
        (this.stats.agent2Rewards.reduce((a, b) => a + b, 0) / this.stats.agent2Rewards.length).toFixed(2) : '0.00',
      epsilon: this.agent2.epsilon.toFixed(3),
      gamesPlayed: this.stats.totalGames
    };
  }
}